(ns adversarial-sketch.leaked-gan
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
            [adversarial-sketch.ground-truth :as ground-truth]
            [adversarial-sketch.image-process :as image-process]))

(def datatype :float)

(def cli-options
  ;; An option with a required argument
  [["s" "--emb-size NUM" "size of embedding data"
    :default 288
    :parse-fn #(Integer/parseInt ^String %)]
   ["m" "--model-type MODEL_TYPE" "bilstm or cnn"
    :default :bilstm
    :parse-fn keyword]
   ["l" "--lstm-size NUM" "lstm size"
    :default 25
    :parse-fn #(Integer/parseInt ^String %)]
   ["-n"  "--num-data DATA"
    :default 100
    :parse-fn #(Integer/parseInt ^String %)]])

;;;;;; One-hot-embeddin init ;;;;;
(def embed-size 288)
(def one-hot-vec (into [] (replicate embed-size 0)) )


;;;;;; This section of code is in charge of random projection of the image data ;;;;;;;;;;;;;;;;;;;;;;;;
(defn random-projection []
  (if (= 0 (rand-int 2)) 1 -1 )
  )


(defn random-proj-multi [observaition random-proj-mat original-dim]
  (let [observ (dge 1 original-dim observaition)]
    (mm observ random-proj-mat))
  )
;;;;;; This section of code is in charge of vectorizing  ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; codes to assign a label to caption of each image individually

(defn vectorize [seq-trace]
  ;; Add the START symbol: 0 and End symbol: embed-size-25.
  ;; We add 24 more symbols after the end for the providing guidance to sketch.
  ;; Then Flattens one level.
  (conj (into [0] (apply concat (map sketch-parser/tracer-vectorizer seq-trace))) (- embed-size 1)))

;;Currently it normalize across the samples, rather than features (row instead of column)
;This may need to be taken care of later.

(defn load-trace-data-target-with-projected-image-withlabel [trace-data-map]
  ;;we may want to append the image symbol and end symbol before flattening
  ;; append the projected image with the next start symbol 0
  (let [orig-dim (* 256 256)
        ki embed-size
        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))]
    (for [k (keys trace-data-map)
          :let [sent (->
                       ; you can The hard coded number n below could give the control over the maximum length
                       ; of the training sequence
                       ;(take-last 5 (vectorize (get trace-data-map k)))
                       (vectorize (get trace-data-map k))
                       ;seq so that the image projection comes first
                       (seq)
                       (conj (-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
                                 (random-proj-multi random-proj-mat orig-dim)
                                 (no/normalize)
                                 )))
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
         :let [e (if (int? word) (assoc one-hot-vec word 1) (vec (apply concat word)))]
         :when (or e unk)]
     (node/const (node/gen-name "word") (or e unk)))))


; The module abstractions

(defn lstm-generator [model ^long lstm-size ^long emb-size]
  (node/let-scope
    ;; let-scope so the parameters get smart-nesting
    ; num_dir 2 gives u bidiretionl. For generating samples, you don't want bidirectional
    [^long num-dirs 2
     input-size (* num-dirs emb-size)
     hidden-size (* num-dirs lstm-size)
     lstm (rnn/lstm-cell model input-size hidden-size)
     hidden->logits (module/affine model emb-size [hidden-size])]
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
                                           (if train? (cg/dropout 0.5 (first output-results))
                                                      (first output-results)))))
                ))
            ;(map #(module/graph hidden->logits %) outputs)
            )))
      ;; build loss node for two-arguments
      (graph-sample [this sent]
        (when-let [inputs (seq (sent-nodes sent))]
          (rnn/sample-seq lstm inputs 5 1 (= num-dirs 2))))
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



(defn get-generator [model-type model lstm-size emb-size]
  (case model-type
    :bilstm (lstm-generator model lstm-size emb-size)))

; MLE training captures the sequences up to the embed-size-25, which is end symbol.
; The one symbol after is not get used in the MLE.
(defn train [{:keys [lstm-size, emb-size, model-type]
              :as opts}]
  (let [train-data (take (:num-data opts) (load-trace-data-target-with-projected-image-withlabel ground-truth/groundTruth))
        ;test-data (take (:num-data opts) (load-trace-data-target-with-projected-image-withlabel ground-truth/groundTruth))
        gen-batches #(partition-all 1 train-data)
        model (model/simple-param-collection)
        ;; classifier can use a cache to avoid
        ;; re-allocating tensors across prediction
        generator (get-generator model-type model lstm-size emb-size)
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
        ;getLogLike-fn (module/getLogLike-fn generator)
        train-opts
        {:num-iters     10
         ;; report train/test accuracy each iter
         :iter-reporters
                        [;; Train Accuracy
                         ;(report/log-likelihood :train (constantly train-data) getLogLike-fn)
                         ;; Test Accuracy
                         ;(report/accuracy :test (constantly test-data) predict-fn)
                         ;; Report performance info on tensor-ops
                         ;(report/callback #(-> (flare/state) :factory flare/debug-info))
                         ]
         :learning-rate 1}
        max_seq_len 1]
    ;; prints shape of all parameters
    (println " Params " (map (juxt first (comp :shape second)) (seq model)))
    (println " Total #params " (model/total-num-params model))
    (train/train! model loss-fn gen-batches train-opts)
    (train/generate! model sample-fn gen-batches train-opts)
    ;(->> input sample-fn (map first) (map :value) (apply (into [])))
    ;(loop [input 0 cnt 0]
    ;  (when (< cnt max_seq_len)
    ;    (print (-> (sample-fn input) count))
    ;    (recur (:value (sample-fn input)) (inc cnt))))
    ))

(defn -main
  " CLI entry point for train/test classifiying a sentence with a single label. See the ns doc for full explanation. "
  [& args]
  (let [parse (parse-opts args cli-options)]
    (println (:options parse))
    (train (:options parse))))

;(comment
;  ;; example of how to run programmatically
;  (doall (load-data " data/sentiment-sentences-train-toy.txt "))
;  (def opts {:embed-file " data/words-to-one-hot-vectors.txt "
;             :lstm-size 25
;             :model-type :lstm
;             :num-data 20
;             :train-file " data/sentiment-sentences-train.txt "
;             :test-file " data/sentiment-sentences-train.txt "
;             :emb-size 1000})
;  ;; Hack to test LSTM end-to-end gradient
;  (do
;    (def emb (load-embeddings opts))
;    (def model (model/simple-param-collection))
;    (def classifier (lstm-sent-lassifier model emb 10 2))
;    (def loss-fn (fn [[sent tag]]
;                   (-> classifier
;                       (with-meta {:train? true})
;                       (module/graph sent tag))))
;    (def df (optimize/loss-fn model loss-fn train-data))
;    (optimize/rand-bump-test df (model/to-doubles model)))
;
;  )


;(defn sparse-projection [d k q]
;  (let [r (multinomial-dist [(/ 1 (* 2 q)) (- 1 (/ 1 q)) (/ 1 (* 2 q))])]
;    (let [indx (.indexOf r 1)]
;      (cond
;        (= indx 0) (Math/sqrt q)
;        (= indx 1) 0
;        :else (* -1 (Math/sqrt q))))))

;(defn random-proj-embedding [observaition d k q]
;  (let [proj (dge d k (take (* d k) (repeatedly (fn [] (sparse-projection d k q)))))
;        observ (dge 1 d observaition)]
;   (mm observ proj))
;  )

;(defn load-trace-data-target-with-image [trace-data-map]
;  (for [k (keys trace-data-map)]
;    [(vectorize (get trace-data-map k))
;     (image-process/png->observation datatype false (str (namespace k) "/" (name k)))]))


;(defn load-trace-data-target-with-embedded-image [trace-data-map d q ki]
;  (for [k (subvec (into [] (keys trace-data-map)) 0 3)]
;    [(vectorize (get trace-data-map k))
;     (-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;         (into [])
;         (random-proj-embedding d ki q))]))


;(defn load-trace-data-target-with-embedded-image [trace-data-map]
;(for [k (subvec (into [] (keys trace-data-map)) 0 3)]
;  [(vectorize (get trace-data-map k))
;   (-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;       (into []))]))


;(defn load-data [path]
;  ;;To have (word, next word) pair as training data
;  ;; Flatten one level
;  (apply concat (for [^String line (line-seq (io/reader path))
;                      :let [sent (.split (.trim line) " ")]]
;                  ;;zip function
;                  (map vector sent (rest sent)))))

;
;(defn load-data-with-label [path,emb]
;  (for [pair (load-data path)]
;    [(first pair) (reduce + (keep-indexed (fn [idx v] (if (pos? v) idx))
;                                          (seq (embeddings/lookup emb (second pair)))))]))

;(defn load-trace-data-target-with-projected-image [trace-data-map]
;  (for [k (keys trace-data-map)]
;    [(vectorize (get trace-data-map k))
;     (-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;         (into [])
;         (random-proj-multi random-proj-mat)
;         )]))


;(defn load-trace-data-target-with-projected-image [trace-data-map]
;  (for [k (keys trace-data-map)]
;    [ [(-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;           (random-proj-multi random-proj-mat)
;           (into [])
;           )
;       0]
;     (vectorize (get trace-data-map k))
;    ]))


;
;(defn load-trace-data-target-with-projected-image [trace-data-map]
;  ;;we may want to append the image symbol and end symbol before flattening
;  ;; append the projected image with the next start sybmol 0
;   (for [k (keys trace-data-map)]
;                  (->
;                    (vectorize (get trace-data-map k))
;                    (seq)
;                    (conj [(-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;                               (random-proj-multi random-proj-mat)
;                               ) 0])
;                    )
;                  ))

;(defn load-trace-data-target-with-projected-image-withlabel [trace-data-map]
;  ;;we may want to append the image symbol and end symbol before flattening
;  ;; append the projected image with the next start sybmol 0
;  (for [k (keys trace-data-map)]
;    (let [sent (->
;                 (vectorize (get trace-data-map k))
;                 (seq)
;                 (conj [(-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;                            (random-proj-multi random-proj-mat)
;                            ) 0])
;                 )
;          ]
;      [sent 0])
;    ))

;
;(defn vectorize [seq-trace]
;  ;;Add the START symbol: 0 and End symbol: 263 and Flattens one level
;  (let [vectorized-data (conj (into [0] (apply concat (map sketch-parser/tracer-vectorizer seq-trace))) 263)]
;    ;;;;zip function to have (word, next word) pair as training trace data
;    (map vector vectorized-data (rest vectorized-data))))
;
;
;
;
;(defn load-trace-data-target-without-projected-image [trace-data-map]
;  ;;apply concat to flat one level
;  ;;we may want to append the image symbol and end symbol before flattening
;  (apply concat (for [k (keys trace-data-map)]
;                  (vectorize (get trace-data-map k)))))
;
;
;
;
;(defn load-falttened-trace-data-target-with-projected-image [trace-data-map]
;  ;;apply concat to flat one level
;  ;;we may want to append the image symbol and end symbol before flattening
;  ;; append the projected image with the next start sybmol 0
;  (apply concat (for [k (keys trace-data-map)]
;                  (->
;                    (vectorize (get trace-data-map k))
;                    (into ())
;                    (conj [(-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;                               (random-proj-multi random-proj-mat)
;                               ) 0])
;                    )
;                  )))

;(defn lstm-generator [model ^long lstm-size ^long emb-size]
;  (node/let-scope
;    ;; let-scope so the parameters get smart-nesting
;    ; num_dir 2 gives u bidiretionl. For generating samples, you don't want bidirectional
;    [^long num-dirs 1
;     input-size (* num-dirs emb-size)
;     hidden-size (* num-dirs lstm-size)
;     lstm (rnn/lstm-cell model input-size hidden-size)
;     hidden->logits (module/affine model emb-size [hidden-size])]
;    (reify
;      module/PModule
;      ;; build logits
;      (graph [this sent]
;        (when-let [inputs (seq (sent-nodes sent))]
;          (let [hiddens (rnn/build-seq lstm inputs (= num-dirs 2))
;                train? (:train? (meta this))
;                output (map first hiddens)
;                ;; take last output as hidden
;                ;output1 (last (map first hiddens))
;                ;output2 (second (map first hiddens))
;                ;output (if train? (cg/dropout 0.5 output) output)
;                ;hidden (last (map first hiddens))
;                ;hidden (if train? (cg/dropout 0.5 hidden) hidden)
;                ]
;            ;(println (cg/multinomial-sample (module/graph hidden->logits output)))
;            ;(println (no/multinomial-dist (seq (:value (module/graph hidden->logits output))) ))
;            ;(module/graph hidden->logits output)
;            (map #(module/graph hidden->logits %) output)
;            )))
;      ;; build loss node for two-arguments
;      (graph-sample [this sent]
;        (when-let [inputs (seq (sent-nodes sent))]
;          (rnn/sample-seq lstm inputs 5 1 (= num-dirs 2))))
;      (graph [this sent label]
;        (when-let [logits (module/graph this sent)]
;          (let [label-node (node/const "label" label)]
;            ;(println (no/multinomial-dist (seq (:value logits)) ))
;            ;(cg/cross-entropy-loss logits label-node)))))))
;            (cg/hadamard logits label-node)))))))


;(defn lstm-generator [model ^long lstm-size ^long emb-size]
;  (node/let-scope
;    ;; let-scope so the parameters get smart-nesting
;    ; num_dir 2 gives u bidiretionl. For generating samples, you don't want bidirectional
;    [^long num-dirs 1
;     input-size (* num-dirs emb-size)
;     hidden-size (* num-dirs lstm-size)
;     lstm (rnn/lstm-cell model input-size hidden-size)
;     hidden->logits (module/affine model emb-size [hidden-size])]
;    (reify
;      module/PModule
;      ;; build logits
;      (graph [this sent]
;        (when-let [inputs (seq (sent-nodes sent))]
;          (let [hiddens (rnn/build-seq lstm inputs (= num-dirs 2))
;                train? (:train? (meta this))
;                ;outputs (map first hiddens)
;                outputs (map first hiddens)
;                ;; take last output as hidden
;                ;output1 (last (map first hiddens))
;                ;output2 (second (map first hiddens))
;                ;output (if train? (cg/dropout 0.5 output) output)
;                ;hidden (last (map first hiddens))
;                ;hidden (if train? (cg/dropout 0.5 hidden) hidden)
;                ]
;            ;(println (cg/multinomial-sample (module/graph hidden->logits output)))
;            ;(println (no/multinomial-dist (seq (:value (module/graph hidden->logits output))) ))
;            ;(module/graph hidden->logits output)
;            (map #(module/graph hidden->logits %) outputs)
;            )))
;      ;; build loss node for two-arguments
;      (graph-sample [this sent]
;        (when-let [inputs (seq (sent-nodes sent))]
;          (rnn/sample-seq lstm inputs 5 1 (= num-dirs 2))))
;      (graph [this sent label]
;        (when-let [logits (module/graph this sent)]
;          (->> (apply map vector (seq [
;                                       (rest (rest logits))
;                                       (map #(node/const "label" [%]) label)
;                                       ]))
;               (map (partial apply cg/cross-entropy-loss))
;               (reduce cg/+)
;               )
;          )))))
;

;(defn lstm-generator [model ^long lstm-size ^long emb-size]
;  (node/let-scope
;    ;; let-scope so the parameters get smart-nesting
;    ; num_dir 2 gives u bidiretionl. For generating samples, you don't want bidirectional
;    [^long num-dirs 1
;     input-size (* num-dirs emb-size)
;     hidden-size (* num-dirs lstm-size)
;     lstm (rnn/lstm-cell model input-size hidden-size)
;     hidden->logits (module/affine model emb-size [hidden-size])]
;    (reify
;      module/PModule
;      ;; build logits
;      (graph [this sent]
;        (when-let [inputs (seq (sent-nodes sent))]
;          (let [hiddens (rnn/build-seq lstm inputs (= num-dirs 2))
;                train? (:train? (meta this))
;                outputs (map first hiddens)
;                ;; take last output as hidden
;                ;output1 (last (map first hiddens))
;                ;output2 (second (map first hiddens))
;                ;output (if train? (cg/dropout 0.5 output) output)
;                ;hidden (last (map first hiddens))
;                ;hidden (if train? (cg/dropout 0.5 hidden) hidden)
;                ]
;            ;(println (cg/multinomial-sample (module/graph hidden->logits output)))
;            ;(println (no/multinomial-dist (seq (:value (module/graph hidden->logits output))) ))
;            ;(module/graph hidden->logits output)
;            (map #(module/graph hidden->logits %) outputs)
;            )))
;      ;; build loss node for two-arguments
;      (graph-sample [this sent]
;        (when-let [inputs (seq (sent-nodes sent))]
;          (rnn/sample-seq lstm inputs 5 1 (= num-dirs 2))))
;      (graph [this sent labels]
;        (when-let [logits (module/graph this sent)]
;          (last (for [logit logits
;                             :let [label-node (node/const "label" [1])]
;                             ]
;                         (cg/cross-entropy-loss logit label-node)))
;          )))))

