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
    :default "/Users/mahdiazarafrooz/Downloads/syntheticTrainingData"
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

;;Currently it normalize across the samples, rather than features (row instead of column)
;This may need to be taken care of later.

;(defn load-trace-data-target-with-projected-image-withlabel [trace-data-map orig-dim]
;  ;;we may want to append the image symbol and end symbol before flattening
;  ;; append the projected image with the next start symbol 0
;  (let [ki embed-size
;        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))]
;    (for [k (keys trace-data-map)
;          :let [sent (->
;                       ; you can The hard coded number n below could give the control over the maximum length
;                       ; of the training sequence
;                       ;(take-last 5 (vectorize (get trace-data-map k)))
;                       (vectorize (get trace-data-map k))
;                       ;seq so that the image projection comes first
;                       (seq)
;                       (conj (-> (image-process/png->observation datatype false (str (namespace k) "/" (name k)))
;                                 (random-proj-multi random-proj-mat orig-dim)
;                                 (no/normalize-vec)
;                                 )))
;                ]]
;      [sent (rest sent)]
;      ))
;  )

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
                       ;(vectorize (take 6 (get trace-data-map (first key-index-pair))))
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





;(defn load-trace-data-target-with-projected-image-withlabel [trace-data-map orig-dim]
;  ;;we may want to append the image symbol and end symbol before flattening
;  ;; append the projected image with the next start symbol 0
;  (let [ki embed-size
;        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))
;        normalized-mat (load-trace-data-target-with-projected-image orig-dim
;                                                     (map #(str (namespace %) "/" (name %)) (keys trace-data-map) ))]
;    (for [k (keys trace-data-map)
;          :let [sent (->
;                       ; you can The hard coded number n below could give the control over the maximum length
;                       ; of the training sequence
;                       ;(take-last 5 (vectorize (get trace-data-map k)))
;                       (vectorize (get trace-data-map k))
;                       ;seq so that the image projection comes first
;                       (seq)
;                       (conj (cols i)))
;                ]]
;      [sent (rest sent)]
;      ))
;  )

;;;;backup for future implementations
;(defn load-trace-data-target-with-projected-image [path-to-files orig-dim]
;  ;;we may want to append the image symbol and end symbol before flattening
;  ;; append the projected image with the next start symbol 0
;  (let [ki embed-size
;        random-proj-mat (dge orig-dim ki (take (* orig-dim ki) (repeatedly (fn [] (random-projection)))))
;        feature-mat (image-process/get-data-matrix path-to-files)]
;    (->> (mapcat #(random-proj-multi % random-proj-mat orig-dim) feature-mat) (apply concat) (vec) (dge 288 100)
;         (no/normalize))
;    )
;
;  )


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
    ;; prints shape of all parameters
    ;(println " Params " (map (juxt first (comp :shape second)) (seq model)))
    ;(println " Total #params " (model/total-num-params model))
    (with-open [is (clojure.java.io/input-stream "lstm-test.edn")]
      (model/from-data! model is))
    ;(train/train! model loss-fn gen-batches train-opts)
    (train/generate! model sample-fn gen-batches train-opts)
    ;(print  (train/generate! model sample-fn gen-batches train-opts))
    ;(let [sketch-output "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*2038552..61488.tmp:257*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2038552..61488.tmp:258\"; //Assume at 2038552..61488.tmp:258\n  assume (shapeIdentity != 0): \"Assume at 2038552..61488.tmp:259\"; //Assume at 2038552..61488.tmp:259\n  assume (shapeIdentity != 1): \"Assume at 2038552..61488.tmp:261\"; //Assume at 2038552..61488.tmp:261\n  int x_s41 = 0;\n  validateX(0, x_s41)//{};\n  int y_s45 = 0;\n  validateY(1, y_s45)//{};\n  int x2_s49 = 0;\n  validateX(2, x2_s49)//{};\n  int y2_s53 = 0;\n  validateY(3, y2_s53)//{};\n  int x_s41_0 = 0;\n  validateX(1, x_s41_0)//{};\n  int y_s45_0 = 0;\n  validateY(2, y_s45_0)//{};\n  int x2_s49_0 = 0;\n  validateX(3, x2_s49_0)//{};\n  int y2_s53_0 = 0;\n  validateY(4, y2_s53_0)//{};\n  _out = (((((shapeIdentity == 2) && (x_s41 == rx1)) && (y_s45 == ry1)) && (x2_s49 == rx2)) && (y2_s53 == ry2)) || (((((shapeIdentity == 2) && (x_s41_0 == rx1)) && (y_s45_0 == ry1)) && (x2_s49_0 == rx2)) && (y2_s53_0 == ry2));\n  return;\n}\n/*2038552..61488.tmp:293*/\n\nvoid renderSpecification (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)/*2038552..61488.tmp:293*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2038552..61488.tmp:294\"; //Assume at 2038552..61488.tmp:294\n  assume (shapeIdentity != 0): \"Assume at 2038552..61488.tmp:295\"; //Assume at 2038552..61488.tmp:295\n  assume (shapeIdentity != 1): \"Assume at 2038552..61488.tmp:297\"; //Assume at 2038552..61488.tmp:297\n  _out = ((((((shapeIdentity == 2) && (0 == rx1)) && (1 == ry1)) && (2 == rx2)) && (3 == ry2)) || (((((shapeIdentity == 2) && (0 == rx1)) && (1 == ry1)) && (2 == rx2)) && (3 == ry2))) || (((((shapeIdentity == 2) && (1 == rx1)) && (2 == ry1)) && (3 == rx2)) && (4 == ry2));\n  return;\n}\n/*2038552..61488.tmp:58*/\n\nvoid validateX (int x, ref int _out)/*2038552..61488.tmp:58*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2038552..61488.tmp:59 (2)\n  assert ((((x == 0) || (x == 1)) || (x == 3)) || (x == 2)); //Assert at 2038552..61488.tmp:60 (2)\n  _out = x;\n  return;\n}\n/*2038552..61488.tmp:53*/\n\nvoid validateY (int x, ref int _out)/*2038552..61488.tmp:53*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2038552..61488.tmp:54 (2)\n  assert ((((x == 1) || (x == 4)) || (x == 3)) || (x == 2)); //Assert at 2038552..61488.tmp:55 (2)\n  _out = x;\n  return;\n}"
    ;      sketch-vector (string/split-lines sketch-output)]
    ;  (print (sketch-parser/parse-sketch-output sketch-vector {} {} 0)))


    ;(-> (train/generate! model sample-fn gen-batches train-opts)
    ;    synthesizer/synthesize-program
    ;    :body
    ;    string/split-lines
    ;    (sketch-parser/parse-sketch-output {} {} 0)
    ;    print)

    ;(->  (:body (synthesizer/synthesize-program [2 6 7 8 9]))
    ;     string/split-lines
    ;     print)
    ;
    ;(let [output (:body (synthesizer/synthesize-program [2 6 7 8 9]))]
    ;  (sketch)
    ;  )
    ;
    ;(sketch-parser/parser )


    ;(->  (:body (synthesizer/synthesize-program [3 6 7 1 2 7 8 9 10]))
    ;     print)

    ;(->  (:body (synthesizer/synthesize-program [3 14 19 1 2 7 8 9 10]))
    ;     string/split-lines
    ;     (sketch-parser/parse-sketch-output {} {} 0)
    ;     print)

    ;
    ;(defn eval-graphic-draw [sketch-ast]
    ;  (->> (-> (sketch-parser/graphic-compiler-transformer sketch-ast)
    ;           string/join
    ;           (str ")))"))
    ;       (str "(defn draw-magic []")
    ;       read-string
    ;       eval))
    ;

    ;(->  (:body (synthesizer/synthesize-program [3 10 10 1]))
    ;     print)


    ;(->   (string/split-lines "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*2114536..25757.tmp:257*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2114536..25757.tmp:258\"; //Assume at 2114536..25757.tmp:258\n  assume (shapeIdentity != 2): \"Assume at 2114536..25757.tmp:260\"; //Assume at 2114536..25757.tmp:260\n  assume (shapeIdentity != 1): \"Assume at 2114536..25757.tmp:261\"; //Assume at 2114536..25757.tmp:261\n  int x_s41 = 0;\n  validateX(1, x_s41)//{};\n  int y_s45 = 0;\n  validateY(1, y_s45)//{};\n  _out = ((shapeIdentity == 0) && (cx == x_s41)) && (cy == y_s45);\n  return;\n}\n/*2114536..25757.tmp:293*/\n\nvoid renderSpecification (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)/*2114536..25757.tmp:293*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2114536..25757.tmp:294\"; //Assume at 2114536..25757.tmp:294\n  assume (shapeIdentity != 2): \"Assume at 2114536..25757.tmp:296\"; //Assume at 2114536..25757.tmp:296\n  assume (shapeIdentity != 1): \"Assume at 2114536..25757.tmp:297\"; //Assume at 2114536..25757.tmp:297\n  _out = ((shapeIdentity == 0) && (cx == 1)) && (cy == 1);\n  return;\n}\n/*2114536..25757.tmp:58*/\n\nvoid validateX (int x, ref int _out)/*2114536..25757.tmp:58*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2114536..25757.tmp:59 (2)\n  assert (x == 1); //Assert at 2114536..25757.tmp:60 (2)\n  _out = x;\n  return;\n}\n/*2114536..25757.tmp:53*/\n\nvoid validateY (int x, ref int _out)/*2114536..25757.tmp:53*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2114536..25757.tmp:54 (2)\n  assert (x == 1); //Assert at 2114536..25757.tmp:55 (2)\n  _out = x;\n  return;\n}\n/* END PACKAGE ANONYMOUS*/\n[SKETCH] DONE")
    ;     (sketch-parser/parse-sketch-output {} {} 0)
    ;     :commands
    ;     print)


    ;(->  (:body (synthesizer/synthesize-program [3 10 10 1]))
    ;     string/split-lines
    ;     (sketch-parser/parse-sketch-output {} {} 0)
    ;     :commands
    ;     print)


    ;(->   (string/split-lines "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpzqJj8W.sk:210\"; //Assume at tmpzqJj8W.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpzqJj8W.sk:212\"; //Assume at tmpzqJj8W.sk:212\n  assume (!(dashed)): \"Assume at tmpzqJj8W.sk:216\"; //Assume at tmpzqJj8W.sk:216\n  assume (!(arrow)): \"Assume at tmpzqJj8W.sk:217\"; //Assume at tmpzqJj8W.sk:217\n  int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)\n        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/\n        {\n          dummyStartBoundary();\n          bit _pac_sc_s26 = _pac_sc_s15_s17_0;\n          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n            _pac_sc_s26 = _pac_sc_s26_s30;\n          }\n          _pac_sc_s15_s17_0 = _pac_sc_s26;\n          dummyEndBoundary();\n        }\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39_1 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);\n          int y_s43_1 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1)")
    ;     (sketch-parser/parse-sketch-output {} {} 0)
    ;     :commands
    ;     print)

    ;
    ;
    ;(defn setup []
    ;  (q/background 255)
    ;  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]))


    ;(defn eval-graphic-draw [sketch-ast]
    ;  (->> (-> (graphic-dsl/graphic-compiler-transformer sketch-ast)
    ;             string/join
    ;             (str ")))"))
    ;         (str "(defn draw-magic []")
    ;         read-string
    ;         eval))

    ;
    ;(->  (:body (synthesizer/synthesize-program [3 10 10 1]))
    ;     string/split-lines
    ;     (sketch-parser/parse-sketch-output {} {} 0)
    ;     :commands
    ;     graphic-dsl/eval-graphic-draw)



    ;(graphic-dsl/run-draw [3 10 20 1]) ;[3 7 7 1 3 8 8 1]
    ;(graphic-dsl/run-draw [2 7 8 9 10])
    ;(graphic-dsl/run-draw [1 7 8 9 10 5 4 1 8 9 10 11 5 4]) ;only 5 4 args worl
    (let [lstm-output [2 6 7 8 9 2 7 8 9 10]
          ast-info (synthesizer/synthesize-program lstm-output)
          ast (:body ast-info)]
      (spit "abstract_synatax_tree.txt" ast-info)
      (graphic-dsl/run-draw-general-ast ast))
      ;(if (string/includes? ast "for(")
      ;  (graphic-dsl/run-draw-ast ast)
      ;  (graphic-dsl/run-draw-loopfree-ast ast)) )

      ;(graphic-dsl/run-draw-ast ast)
      ;(graphic-dsl/run-draw-loopfree-ast ast)





    ;;(def sketch-output "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*2316715..47823.tmp:251*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2316715..47823.tmp:252\"; //Assume at 2316715..47823.tmp:252\n  assume (shapeIdentity != 2): \"Assume at 2316715..47823.tmp:254\"; //Assume at 2316715..47823.tmp:254\n  assume (shapeIdentity != 1): \"Assume at 2316715..47823.tmp:255\"; //Assume at 2316715..47823.tmp:255\n  int x_s41 = 0;\n  validateX(1, x_s41)//{};\n  int y_s45 = 0;\n  validateY(1, y_s45)//{};\n  int x_s41_0 = 0;\n  validateX(2, x_s41_0)//{};\n  int y_s45_0 = 0;\n  validateY(2, y_s45_0)//{};\n  _out = (((shapeIdentity == 0) && (cx == x_s41)) && (cy == y_s45)) || (((shapeIdentity == 0) && (cx == x_s41_0)) && (cy == y_s45_0));\n  return;\n}\n/*2316715..47823.tmp:280*/\n\nvoid renderSpecification (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)/*2316715..47823.tmp:280*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2316715..47823.tmp:281\"; //Assume at 2316715..47823.tmp:281\n  assume (shapeIdentity != 2): \"Assume at 2316715..47823.tmp:283\"; //Assume at 2316715..47823.tmp:283\n  assume (shapeIdentity != 1): \"Assume at 2316715..47823.tmp:284\"; //Assume at 2316715..47823.tmp:284\n  _out = (((shapeIdentity == 0) && (cx == 1)) && (cy == 1)) || (((shapeIdentity == 0) && (cx == 2)) && (cy == 2));\n  return;\n}\n/*2316715..47823.tmp:58*/\n\nvoid validateX (int x, ref int _out)/*2316715..47823.tmp:58*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2316715..47823.tmp:59 (2)\n  assert ((x == 1) || (x == 2)); //Assert at 2316715..47823.tmp:60 (2)\n  _out = x;\n  return;\n}\n/*2316715..47823.tmp:53*/\n\nvoid validateY (int x, ref int _out)/*2316715..47823.tmp:53*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2316715..47823.tmp:54 (2)\n  assert ((x == 1) || (x == 2)); //Assert at 2316715..47823.tmp:55 (2)\n  _out = x;\n  return;\n}")
    ;(let [sketch-ast "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpzqJj8W.sk:210\"; //Assume at tmpzqJj8W.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpzqJj8W.sk:212\"; //Assume at tmpzqJj8W.sk:212\n  assume (!(dashed)): \"Assume at tmpzqJj8W.sk:216\"; //Assume at tmpzqJj8W.sk:216\n  assume (!(arrow)): \"Assume at tmpzqJj8W.sk:217\"; //Assume at tmpzqJj8W.sk:217\n  int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)\n        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/\n        {\n          dummyStartBoundary();\n          bit _pac_sc_s26 = _pac_sc_s15_s17_0;\n          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n            _pac_sc_s26 = _pac_sc_s26_s30;\n          }\n          _pac_sc_s15_s17_0 = _pac_sc_s26;\n          dummyEndBoundary();\n        }\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39_1 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);\n          int y_s43_1 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1)"]
    ;  (graphic-dsl/run-draw-general-ast sketch-ast))

    ;(let [lstm-output [2 6 7 8 9 2 7 8 9 10]]
    ;  (print (synthesizer/synthesize-program lstm-output)))


    ;(->   (string/split
    ;        "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*2202058..94584.tmp:257*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2202058..94584.tmp:258\"; //Assume at 2202058..94584.tmp:258\n  assume (shapeIdentity != 0): \"Assume at 2202058..94584.tmp:259\"; //Assume at 2202058..94584.tmp:259\n  assume (shapeIdentity != 1): \"Assume at 2202058..94584.tmp:261\"; //Assume at 2202058..94584.tmp:261\n  int x_s41 = 0;\n  validateX(0, x_s41)//{};\n  int y_s45 = 0;\n  validateY(1, y_s45)//{};\n  int x2_s49 = 0;\n  validateX(2, x2_s49)//{};\n  int y2_s53 = 0;\n  validateY(3, y2_s53)//{};\n  int x_s41_0 = 0;\n  validateX(1, x_s41_0)//{};\n  int y_s45_0 = 0;\n  validateY(2, y_s45_0)//{};\n  int x2_s49_0 = 0;\n  validateX(3, x2_s49_0)//{};\n  int y2_s53_0 = 0;\n  validateY(4, y2_s53_0)//{};\n  _out = (((((shapeIdentity == 2) && (x_s41 == rx1)) && (y_s45 == ry1)) && (x2_s49 == rx2)) && (y2_s53 == ry2)) || (((((shapeIdentity == 2) && (x_s41_0 == rx1)) && (y_s45_0 == ry1)) && (x2_s49_0 == rx2)) && (y2_s53_0 == ry2));\n  return;\n}\n/*2202058..94584.tmp:292*/\n\nvoid renderSpecification (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)/*2202058..94584.tmp:292*/\n{\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at 2202058..94584.tmp:293\"; //Assume at 2202058..94584.tmp:293\n  assume (shapeIdentity != 0): \"Assume at 2202058..94584.tmp:294\"; //Assume at 2202058..94584.tmp:294\n  assume (shapeIdentity != 1): \"Assume at 2202058..94584.tmp:296\"; //Assume at 2202058..94584.tmp:296\n  _out = (((((shapeIdentity == 2) && (0 == rx1)) && (1 == ry1)) && (2 == rx2)) && (3 == ry2)) || (((((shapeIdentity == 2) && (1 == rx1)) && (2 == ry1)) && (3 == rx2)) && (4 == ry2));\n  return;\n}\n/*2202058..94584.tmp:58*/\n\nvoid validateX (int x, ref int _out)/*2202058..94584.tmp:58*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2202058..94584.tmp:59 (2)\n  assert ((((x == 0) || (x == 1)) || (x == 3)) || (x == 2)); //Assert at 2202058..94584.tmp:60 (2)\n  _out = x;\n  return;\n}\n/*2202058..94584.tmp:53*/\n\nvoid validateY (int x, ref int _out)/*2202058..94584.tmp:53*/\n{\n  assert ((x >= 0) && (x <= 16)); //Assert at 2202058..94584.tmp:54 (2)\n  assert ((((x == 1) || (x == 4)) || (x == 3)) || (x == 2)); //Assert at 2202058..94584.tmp:55 (2)\n  _out = x;\n  return;"
    ;        #"[\n\|\|]")
    ;      (sketch-parser/parse-sketch-output {} {} 0)
    ;     :commands
    ;     print)




    ;(declare draw-magic)
    ;
    ;(q/defsketch trigonometry
    ;             :title "compiled codes"
    ;             :setup setup
    ;             :size [300 300]
    ;             :draw draw-magic)

    ;(q/defsketch trigonometry
    ;             :title "compiled codes"
    ;             :setup dsl/setup
    ;             :size [300 300]
    ;             :draw dsl/draw-test2)


    ;(dsl/dsl-draw)
    ;; run sketch
    ;(q/defsketch trigonometry
    ;             :size [300 300]
    ;;             :draw draw-new)

   ;(print  (sketch-parser/parse-sketch-output
   ;          (string/split-lines (str (:body (synthesizer/synthesize-program [2 6 7 8 9 2 7 8 9 10])))) {} {} 0))
    ;(print (->  (synthesizer/synthesize-program [2 6 7 8 9 2 7 8 9 10])
    ;            string/split-lines))
    ;(print (str (synthesizer/synthesize-program [2 6 7 8 9 2 7 8 9 10])))

    ;(with-open [os (clojure.java.io/output-stream "lstm-test.edn")]
    ;  (model/to-data! model os ))

    ;(->> input sample-fn (map first) (map :value) (apply (into [])))
    ;(loop [input [0] cnt 0]
    ;  (when (< cnt max_seq_len)
    ;    (print (-> (sample-fn input 5 1) count))
    ;    (recur (:value (sample-fn input)) (inc cnt))))
    ))

(defn -main
  " CLI entry point for train/test classifiying a sentence with a single label. See the ns doc for full explanation. "
  [& args]
  (let [parse (parse-opts args cli-options)]
    ;(println (:options parse))
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
