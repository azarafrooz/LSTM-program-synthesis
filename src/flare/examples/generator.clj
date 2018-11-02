(ns flare.examples.generator
  (:require [flare.node :as node]
            [flare.cnn :as cnn]
            [flare.rnn :as rnn]
            [flare.compute :as compute]
            [flare.embeddings :as embeddings]
            [clojure.java.io :as io]
            [clojure.tools.cli :refer [parse-opts]]
            [flare.neanderthal-ops :as no]
            [flare.model :as model]
            [flare.report :as report]
            [flare.computation-graph :as cg]
            [flare.train :as train]
            [flare.module :as module]
            [flare.optimize :as optimize]
            [flare.core :as flare]))

(def cli-options
  ;; An option with a required argument
  [["-train" "--train-file PATH" "path to data"
    :default "data/sentiment-sentences-train.txt"]
   ["-test" "--test-file PATH" "path to test"
    :default "data/sentiment-sentences-test.txt"]
   ["-e" "--embed-file PATH" "path to data"
    :default "data/words-to-one-hot-vectors.txt"]
   ["s" "--emb-size NUM" "size of embedding data"
    :default 1000
    :parse-fn #(Integer/parseInt ^String %)]
   ["m" "--model-type MODEL_TYPE" "bilstm or cnn"
    :default :bilstm
    :parse-fn keyword]
   ["l" "--lstm-size NUM" "lstm size"
    :default 25
    :parse-fn #(Integer/parseInt ^String %)]
   ["-n"  "--num-data DATA"
    :default 10
    :parse-fn #(Integer/parseInt ^String %)]])

(defn load-embeddings [opts]
  (embeddings/fixed-embedding
    (no/factory)
    (:emb-size opts)
    (-> opts :embed-file io/reader embeddings/read-text-embedding-pairs)))

(flare/init!)

(defn lstm-generator [model word-emb ^long lstm-size ^long emb-size]
  (node/let-scope
    ;; let-scope so the parameters get smart-nesting
    [^long emb-size (embeddings/embedding-size word-emb)
     ; num_dir 2 gives u bidiretionl. For generating samples, you don't want bidirectional
     ^long num-dirs 1
     input-size (* num-dirs emb-size)
     hidden-size (* num-dirs lstm-size)
     lstm (rnn/lstm-cell model input-size hidden-size)
     hidden->logits (module/affine model emb-size [hidden-size])]
    (reify
      module/PModule
      ;; build logits
      (graph [this sent]
        (when-let [inputs (seq (embeddings/sent-nodes word-emb (vector sent)))]
          (let [hiddens (rnn/build-seq lstm inputs (= num-dirs 2))
                train? (:train? (meta this))
                ;; take last output as hidden
                output (last (map first hiddens))
                output (if train? (cg/dropout 0.5 output) output)
                ;hidden (last (map first hiddens))
                ;hidden (if train? (cg/dropout 0.5 hidden) hidden)
                ]
            ;(println (cg/multinomial-sample (module/graph hidden->logits output)))
            ;(println (no/multinomial-dist (seq (:value (module/graph hidden->logits output))) ))
            (module/graph hidden->logits output))))
      ;; build loss node for two-arguments
      (graph-sample [this sent]
        (when-let [inputs (seq (embeddings/sent-nodes word-emb (vector sent)))]
          (rnn/sample-seq lstm inputs 5 1 (= num-dirs 2))))
      (graph [this sent label]
        (when-let [logits (module/graph this sent)]
          (let [label-node (node/const "label" [label])]
            ;(println (no/multinomial-dist (seq (:value logits)) ))
            (cg/cross-entropy-loss logits label-node)))))))


(defn load-data [path]
  ;;To have (word, next word) pair as training data
  ;; Flatten one level
  (apply concat (for [^String line (line-seq (io/reader path))
                      :let [sent (.split (.trim line) " ")]]
                  ;;zip function
                  (map vector sent (rest sent)))))

(defn load-data-with-label [path,emb]
  (for [pair (load-data path)]
    [(first pair) (reduce + (keep-indexed (fn [idx v] (if (pos? v) idx))
               (seq (embeddings/lookup emb (second pair)))))]))

(defn get-generator [model-type model emb lstm-size emb-size]
  (case model-type
    :bilstm (lstm-generator model emb lstm-size emb-size)))


(defn train [{:keys [lstm-size, emb-size, train-file, test-file model-type]
              :as opts}]
  (let [emb (load-embeddings opts)
        start-data (into [] (replicate emb-size 0))
        train-data (take (:num-data opts) (load-data-with-label train-file emb))
        test-data (take (:num-data opts) (load-data-with-label test-file emb))
        gen-batches #(partition-all 32 train-data)
        model (model/simple-param-collection)
        ;; classifier can use a cache to avoid
        ;; re-allocating tensors across prediction
        generator (get-generator model-type model emb lstm-size emb-size)
        ;; pair is [word nextword]
        loss-fn (fn [[sent tag]]
                  (-> generator
                      (with-meta {:train? true})
                      (module/graph sent tag)))
        sample-fn (fn [sent]
                  (-> generator
                      (with-meta {:train? false})
                      (module/graph-sample sent)))
        predict-fn (module/predict-fn generator)
        train-opts
        {:num-iters 2
         ;; report train/test accuracy each iter
         :iter-reporters
                    [;; Train Accuracy
                     (report/accuracy :train (constantly train-data) predict-fn)
                     ;; Test Accuracy
                     (report/accuracy :test (constantly test-data) predict-fn)
                     ;; Report performance info on tensor-ops
                     (report/callback #(-> (flare/state) :factory flare/debug-info))]
         :learning-rate 1}
        max_seq_len 1]
    ;; prints shape of all parameters
    (println "Params " (map (juxt first (comp :shape second)) (seq model)))
    (println "Total # params " (model/total-num-params model))
    (train/train! model loss-fn gen-batches train-opts)
    ;(->> input sample-fn (map first) (map :value) (apply (into [])))
    (loop [input "under" cnt 0]
      (when (< cnt max_seq_len)
        (print (-> (sample-fn input) count))
        (recur (:value (sample-fn input)) (inc cnt))))
    ))

(defn -main
  "CLI entry point for train/test classifiying a sentence with a single label. See the ns doc for full explanation."
  [& args]
  (let [parse (parse-opts args cli-options)]
    (println (:options parse))
    (train (:options parse))))

(comment
  ;; example of how to run programmatically
  (doall (load-data "data/sentiment-sentences-train-toy.txt"))
  (def opts {:embed-file "data/words-to-one-hot-vectors.txt"
             :lstm-size 25
             :model-type :lstm
             :num-data 20
             :train-file "data/sentiment-sentences-train.txt"
             :test-file "data/sentiment-sentences-train.txt"
             :emb-size 1000})
  ;; Hack to test LSTM end-to-end gradient
  (do
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection))
    (def classifier (lstm-sent-lassifier model emb 10 2))
    (def loss-fn (fn [[sent tag]]
                   (-> classifier
                       (with-meta {:train? true})
                       (module/graph sent tag))))
    (def df (optimize/loss-fn model loss-fn train-data))
    (optimize/rand-bump-test df (model/to-doubles model)))

  )
