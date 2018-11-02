(ns adversarial-sketch.rp-lstm
  (:use [uncomplicate.neanderthal core native])
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
            [flare.core :as flare]
            [adversarial-sketch.sketch-parser :as sketch-parser]
            [adversarial-sketch.graphical-language :as graphic-dsl]
            [adversarial-sketch.ground-truth :as ground-truth]
            [adversarial-sketch.image-process :as image-process]
            [adversarial-sketch.config :as config]
            [adversarial-sketch.core :as synthesizer]
            [quil.core :as q :include-macros true]
            [clojure.string :as string]))
(def datatype :float)
(def cli-options
  ;; An option with a required argument
  [["s" "--emb-size NUM" "size of embedding data"
    :default config/embed-size
    :parse-fn #(Integer/parseInt ^String %)]
   ["m" "--model-type MODEL_TYPE" "bilstm or cnn"
    :default :bilstm
    :parse-fn keyword]
   ["l" "--lstm-size NUM" "lstm size"
    :default config/lstm-size
    :parse-fn #(Integer/parseInt ^String %)]
   ["n"  "--num-data DATA"
    :default config/num-data
    :parse-fn #(Integer/parseInt ^String %)]
   ["max_seq_length"  "--max_seq_length"
    :default config/max_seq_length
    :parse-fn #(Integer/parseInt ^String %)]
   ["num"  "--num_samples"
    :default 1
    :parse-fn #(Integer/parseInt ^String %)]
   ["d"  "--orig-dim"
    :default (* 256 256)
    :parse-fn #(Integer/parseInt ^String %)]
   ["train-data-dir"  "--train-data-dir"
    :default "syntheticTrainingData"
    :parse-fn #(str %)]
   ])


;;;;;; One-hot-embeddin init ;;;;;
(def embed-size config/embed-size)
(def guidance-token-size config/guidance-token-size)
(def one-hot-vec (into [] (replicate embed-size 0)) )


;;;;;; This section of code is in charge of random projection of the image data ;;;;;;;;;;;;;;;;;;;;;;;;
(defn random-projection []
  (if (= 0 (rand-int 2)) 1 -1 )
  )


(defn random-proj-multi [observation random-proj-mat original-dim]
  (let [observ (dge 1 original-dim observation)]
    (mm observ random-proj-mat)))
;;;;;; This section of code is in charge of vectorizing  ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; codes to assign a label to caption of each image individually

(defn vectorize [seq-trace]
  ;; Add the START symbol: 0 and End symbol: embed-size-25.
  ;; We add 24 more symbols after the end for the providing guidance to sketch.
  ;; Then Flattens one level.
  (conj (into [0] (apply concat (map sketch-parser/tracer-vectorizer seq-trace))) (- embed-size (+ 1 guidance-token-size))))



;;;;;;;;;;; loadDatawithLabel With new normalization
;;;;;;;;;;;;;; Loading, random projecting and normalizing of the image

(defn load-trace-data-target-with-projected-image [path-to-files num-data orig-dim]
  ;;we may want to append the image symbol and end symbol before flattening
  ;; append the projected image with the next start symbol 0
  (let [ki embed-size
        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))
        feature-mat (map #(image-process/png->observation :float false %) path-to-files)]
    (->> (mapcat #(random-proj-multi % random-proj-mat orig-dim) feature-mat)
         (apply concat)
         (vec)
         (dge embed-size num-data)
         (no/normalize)
         )
    )

  )



(defn load-trace-data-target-with-projected-image-withlabel-train [trace-data-map train-data-dir num-data orig-dim]
  ;;we may want to append the image symbol and end symbol before flattening
  ;; append the projected image with the next start symbol 0
  (let [trace-data-map (into {} (take num-data trace-data-map))
        ki embed-size
        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))
        normalized-mat
        (load-trace-data-target-with-projected-image
          (map #(str train-data-dir "/" (name %)) (keys trace-data-map) ) num-data orig-dim
          )]
    (for [key-index-pair (apply map vector [(keys trace-data-map) (range (count trace-data-map))])
          :let [sent (->
                       (vectorize (get trace-data-map (first key-index-pair)))
                       ;seq so that the image projection comes first
                       (seq)
                       (conj (col normalized-mat (second key-index-pair))))
                ]]
      [sent (rest sent)]
      ))
  )



(defn load-trace-data-target-with-projected-image-withlabel-test [trace-data-map num-data orig-dim]
  ;;we may want to append the image symbol and end symbol before flattening
  ;; append the projected image with the next start symbol 0
  (let [ki embed-size
        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))
        normalized-mat
        (load-trace-data-target-with-projected-image
          (map #(str (namespace %) "/" (name %)) (keys trace-data-map) ) num-data orig-dim
                                                                    )]
    (for [key-index-pair (apply map vector [(keys trace-data-map) (range (count trace-data-map))])
          :let [sent (->
                       (vectorize (get trace-data-map (first key-index-pair)))
                       ;seq so that the image projection comes first
                       (seq)
                       (conj (col normalized-mat (second key-index-pair))))
                ]]
      [sent (rest sent)]
      ))
  )







;;;;;  Initiating the flare ;;;;;;;;;;;;;;;;;;;;;;;

(flare/init!)

;;;;; This section of code is in charge of preparing the data for lstm nodes ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn sent-nodes
  "take a sentence and return sequence of constant node
   tensors, where each constant has the original word
   as part of the name.

   Will use `unk` if given for unknown tokens, or omit
   if `unk` isn't passed in"
  ([sent] (sent-nodes sent nil))
  ([sent unk]
   (for [word sent
         ;Check to see if the word is already a embedded vec (as in projected image) or not.
         :let [e (if (int? word) (assoc one-hot-vec word 1) (vec (seq word)))]
         :when (or e unk)]
     (node/const (node/gen-name "word") (or e unk)))))


; The module abstractions

(defn lstm-generator [model ^long lstm-size ^long emb-size max_seq_length]
  (node/let-scope
    ;; let-scope so the parameters get smart-nesting
    ; num_dir 2 gives u bidiretionl. For generating samples, you don't want bidirectional
    [^long num-dirs 1
     input-size (* num-dirs emb-size)
     hidden-size (* num-dirs lstm-size)
     lstm (rnn/lstm-cell model input-size hidden-size)
     hidden->logits (module/affine model emb-size [hidden-size])
     end_symbol (- embed-size (+ guidance-token-size 1))
     start_symbol 0
     num_samples 1]
    (reify
      module/PModule
      ;; build logits
      (graph [this sent]
        (when-let [inputs (seq (sent-nodes sent))]
          (let [hiddens (rnn/build-seq lstm inputs (= num-dirs 2))
                train? (:train? (meta this))
                outputs (map first hiddens)
                ]
            ;The following block of code and the line after seems to work better memory wise.
            (loop [output-results outputs output-nodes []]
              (if (= (count output-results) 1)
                output-nodes
                (recur (next output-results)
                       (conj output-nodes
                             (module/graph hidden->logits
                                           (if train? (cg/dropout config/dropout (first output-results))
                                                      (first output-results)))))
                ))
            ;(map #(module/graph hidden->logits %) outputs)
            )))
      ;; generates samples give an image, the first input is the image, the second is the start symbol.
      ;start_symbol is 0 and end_symbol is embed-size-guidance-token-size-1
      (graph-sample [this word]
        (when-let [inputs (seq (sent-nodes word))]
          (rnn/sample-seq lstm hidden->logits inputs max_seq_length num_samples start_symbol end_symbol)))
      (graph [this sent labels]
        (when-let [logits (module/graph this sent)]
          ;rest logits to skip the observation of the projected image.
          (loop [logit-seq (rest logits) label-seq labels comp-result []]
            (let [logit (first logit-seq)
                  label-node (node/const "label" [(first label-seq)])]
              (if (= (count label-seq) 1)
                (cg/scale (/ 1 (count sent)) (reduce cg/+ comp-result))
                (recur (next logit-seq) (next label-seq) (conj comp-result (cg/cross-entropy-loss logit label-node))))))

          )))))



(defn get-generator [model-type model lstm-size emb-size max_seq_length]
  (case model-type
    :bilstm (lstm-generator model lstm-size emb-size max_seq_length)))

; MLE training captures the sequences up to the embed-size-25, which is end symbol.
; The one symbol after is not get used in the MLE.
(defn train [{:keys [lstm-size, emb-size, model-type, max_seq_length, num_samples, start_symbol, end_symbol, orig-dim, train-data-dir]
              :as opts}]
  (let [train-data (take (:num-data opts) (load-trace-data-target-with-projected-image-withlabel-train ground-truth/all-records train-data-dir (:num-data opts) orig-dim))
        ;test-data (take (:num-data opts) (load-trace-data-target-with-projected-image-withlabel-test ground-truth/groundTruth num-data orig-dim))
        gen-batches #(partition-all config/batch-size train-data)
        model (model/simple-param-collection)
        ;; classifier can use a cache to avoid
        ;; re-allocating tensors across prediction
        generator (get-generator model-type model lstm-size emb-size max_seq_length)
        ;; pair is [word nextword]
        loss-fn (fn [[sent tag]]
                  (-> generator
                      (with-meta {:train? true})
                      (module/graph sent tag)))
        sample-fn (fn [word max_seq_length num_samples]
                    (-> generator
                        (with-meta {:train? false})
                        (module/graph-sample word)))
        predict-fn (module/predict-fn generator)
        ;getLogLike-fn (module/getLogLike-fn generator)
        train-opts
        {:num-iters     config/num-iters
         ;; report train/test accuracy each iter
         :iter-reporters
                        [;; Train Accuracy
                         ;(report/log-likelihood :train (constantly train-data) getLogLike-fn)
                         ;; Test Accuracy
                         ;(report/accuracy :test (constantly test-data) predict-fn)
                         ;; Report performance info on tensor-ops
                         ;(report/callback #(-> (flare/state) :factory flare/debug-info))
                         ]
         :learning-rate config/learning-rate}
        max_seq_len 1]
   
    (with-open [is (clojure.java.io/input-stream "lstm.edn")]
      (model/from-data! model is))
    (train/generate! model sample-fn gen-batches train-opts)
    
    (let [lstm-output [2 6 7 8 9 2 7 8 9 10]
          ast-info (synthesizer/synthesize-program lstm-output)
          ast (:body ast-info)]
      (spit "abstract_synatax_tree.txt" ast-info)
      (graphic-dsl/run-draw-general-ast ast))

    ))

(defn -main
  " CLI entry point for train/test classifiying a sentence with a single label. See the ns doc for full explanation. "
  [& args]
  (let [parse (parse-opts args cli-options)]
    (train (:options parse))))
