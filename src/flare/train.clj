(ns flare.train
  (:require
   [clojure.pprint :as pprint]
   [clojure.spec.alpha :as s]
   [flare.model :as model]
   [flare.compute :as compute]
   [flare.core :as flare]
   [flare.report :as report]
   [flare.optimize :as optimize]
   [flare.graph :as graph]
   [flare.module :as module]))

(s/def ::train-opts
  (s/keys :opt-un
          [::num-iters, ::optimizer, ::learning-rate, ::iter-reporters]))

(s/def ::num-iters int?)
(s/def ::optimizer (partial satisfies? optimize/Optimizer))
(s/def ::learning-rate number?)
(s/def ::iter-reporters (s/coll-of (partial satisfies? report/Reporter)))

(def +default-train-opts+
  {:num-iters 100
   :learning-rate 0.01})

(defn run-batch!
  "run forward/backward passes for each of the loss graphs created
   by `loss-module` on the batch.

   Each entry in batch as treated as arguments to `module/graph`
   to build an expression for example loss"
  [factory get-loss-node batch]
  (let [{:keys [eager?, cache]} (flare/state)]
    (loop [batch-loss 0.0 batch batch]
      (if-let [data (first batch)]
        (if-let [loss-node (get-loss-node data)]
          (let [loss-node (if eager?
                            loss-node
                            (compute/forward-pass! factory loss-node cache))
                _ (flare/copy! (:grad loss-node) [1.0])
                loss-val (-> loss-node :value seq first)]
            ;; side-effect to update gradients
            (compute/backward-pass! factory loss-node cache)
            (recur (+ batch-loss 0.0 (double loss-val)) (next batch)))
          (recur batch-loss (next batch)))
        batch-loss))))

(defn iter! [model optimizer get-loss-node data-gen opts]
  (let [total-loss (atom 0.0)
        factory (model/tensor-factory model)]
    (doseq [batch (data-gen)]
      (optimize/reset-batch! optimizer model)
      (let [batch-loss (run-batch! factory get-loss-node batch)]
        (swap! total-loss + batch-loss))
      (optimize/update-model! optimizer model))
    @total-loss))


(defn train!
  "Trains a model using the following parameters:

  * `model` a `flare.model/PModel` instance
  * `get-loss-node` a function which takes each data input
    and returns a node representing the loss in the input
  * `data-batch-fn` assumed that calling this function yields
    a lazy sequence of data batches, looping over each batch
    and calls `get-loss-node`
  * `train-opts`, see spec above"
  ([model get-loss-node data-batch-fn train-opts]
   (s/assert ::train-opts train-opts)
   (let [factory (model/tensor-factory model)
         opts (merge +default-train-opts+ train-opts)
         optimizer (:optimizer train-opts
                               (optimize/->Adadelta factory 1.0 0.9 1e-6))]
     (println "Optimizing with" (type optimizer))
     (optimize/init-model! optimizer model)
     (dotimes [iter (:num-iters train-opts)]
       (doseq [reporter (:iter-reporters train-opts)]
         (report/clear! reporter))
       (printf "Iteration %d\n" iter)
       (let [time (System/currentTimeMillis)
             loss (iter! model optimizer get-loss-node data-batch-fn train-opts)]
         (doseq [reporter (:iter-reporters train-opts)]
           (when-let [r (report/gen reporter)]
             (pprint/pprint r)))
         (let [delta-ms (- (System/currentTimeMillis) time)]
           (printf "End of iteration %d: %.3f (%d ms) \n" iter loss delta-ms)
           (.flush System/out))))))
  ([model get-loss-node data-batch-fn]
   (train! model get-loss-node data-batch-fn {})))


;;;;;; The codes required for the generating samples

(defn gen-run-batch!
  "run sample generation from graphs created
   by `generate-module` on the batch.

   Each entry in batch as treated as arguments to `module/graph`
   to build an expression for example loss"
  [factory sample-fn max_seq_length num_samples batch]
  (let [{:keys [eager?, cache]} (flare/state)]
    (loop [outputs [] batch batch]
      (if-let [data (first batch)]
        (if-let [generated-seq (sample-fn [(first (first data))] max_seq_length num_samples)]
          (recur (conj outputs generated-seq) (next batch)))
        outputs))))


(defn gen-iter! [model sample-fn data-gen opts]
  (let [all-samples (transient [])
        factory (model/tensor-factory model)
        max_seq_length (:max_seq_length opts)
        num_samples (:num_samples opts)]
    (doseq [batch (data-gen)]
      (let [samples (gen-run-batch! factory sample-fn max_seq_length num_samples batch)]
        (conj! all-samples samples)))
   (persistent! all-samples)))


;;;samples (gen-run-batch! factory sample-fn max_seq_length num_samples batch)



(defn generate!
  "Trains a model using the following parameters:

  * `model` a `flare.model/PModel` instance
  * `sample-fn` a function which takes each data input
    and returns a sequence of generated output
  * `data-batch-fn` assumed that calling this function yields
    a lazy sequence of data batches, looping over each batch
    and calls `sample-fn`
  * `train-opts`, see spec above"
  ([model sample-fn data-batch-fn train-opts]
   (s/assert ::train-opts train-opts)
   (let [factory (model/tensor-factory model)
         opts (merge +default-train-opts+ train-opts)]
     (dotimes [iter 1]
       (doseq [reporter (:iter-reporters train-opts)]
         (report/clear! reporter))
       (printf "Iteration %d\n" iter)
       (let [time (System/currentTimeMillis)
             samples (gen-iter! model sample-fn data-batch-fn train-opts)]
         (doseq [reporter (:iter-reporters train-opts)]
           (when-let [r (report/gen reporter)]
             (pprint/pprint r)))
         (let [delta-ms (- (System/currentTimeMillis) time)]
           (print samples)
           ;samples
           ;(printf "End of iteration %d: %.3f (%d ms) \n" iter samples delta-ms)
           (.flush System/out))))))
  ([model sample-fn data-batch-fn]
   (generate! model sample-fn data-batch-fn {})))
