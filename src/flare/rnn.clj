(ns flare.rnn
  (:require [flare.model :as model]
            [flare.core :as flare]
            [flare.node :as node]
            [flare.computation-graph :as cg]
            [flare.module :as module]
            [flare.neanderthal-ops :as no]
            [adversarial-sketch.config :as config])
  (:import [flare.node Node]))

(defprotocol RNNCell
  (cell-model [this]
    "return model underlying cell")
  (output-dim [this]
    "output dimension of cell")
  (input-dim [this]
    "dimension of inputs")
  (init-hidden [this]
    "return initial (output, cell-state) pair")
  (add-input [this input last-hidden]
    "take last hidden and reutrn new hidden"))

(defn lstm-cell
  "Standard LSTM cell 
    https://en.wikipedia.org/wiki/Long_short-term_memory
  without any peepholes or other adaptations.

   The hidden state is a pair (output, cell-state)"
  [model ^long input-dim ^long hidden-dim]
  (node/let-scope
      [;; concatenate previous output and cur input
       cat-dim (+ input-dim hidden-dim)
       ;; stack (input, output, forget, state) params
       ;; one affine module W_(i,o,f,s) * x_(prev, input) + b_(i,o,f,s)
       input->gates (module/affine model (* 4 hidden-dim) [cat-dim])
       zero  (flare/zeros [hidden-dim])
       init-output (node/const "h0" zero)
       init-state (node/const "c0"  zero)]
    (reify RNNCell
      (cell-model [this] model)
      (output-dim [this] hidden-dim)
      (input-dim [this] input-dim)
      (init-hidden [this] [init-output init-state])
      (add-input [this input [last-output last-state]]
        (flare/validate-shape! [input-dim] (:shape input))
        (flare/validate-shape! [hidden-dim] (:shape last-state))
        (let [x (cg/concat 0 input last-output)
              gates (module/graph input->gates x)
              ;; split (i,o,f) and state
              [iof, state] (cg/split gates 0 (* 3 hidden-dim))
              ;; one big sigmloid then split into (input, forget, output)
              [input-probs forget-probs output-probs]
                (cg/split (cg/sigmoid iof) 0 hidden-dim (* 2 hidden-dim))
              ;; combine hadamard of forget past, keep present
              state (cg/+
                     (cg/hadamard forget-probs last-state)
                     (cg/hadamard input-probs (cg/tanh state)))
              output (cg/hadamard output-probs (cg/tanh state))]
          [output state])))))


(defn build-seq
  "return sequence of `add-input` outputs for a given `RNNCell`
  and sequence of `inputs`. Can optionally make sequence building
  bidrectional using `bidirectional?` optional third argument.

  Returned sequence will drop the `init-hidden` element which doesn't
  correspond to an input."
  ([cell inputs] (build-seq cell inputs false))
  ([cell inputs bidrectional?]
   (let [factory (-> cell cell-model model/tensor-factory)
         out-dim (output-dim cell)
         hidden (init-hidden cell)
         ;; for bidirectional, concat reversed version of input
         inputs (if bidrectional?
                  (map #(cg/concat 0 %1 %2) inputs (reverse inputs))
                  inputs)]
     (loop [inputs inputs hiddens (list (init-hidden cell))]
       (if-let [input (first inputs)]
         (let [last-hidden (first hiddens)
               hidden (add-input cell input last-hidden)]
           (recur (next inputs) (cons hidden hiddens)))
         ;; states/outputs are built in reverse and initial state is
         ;; just so the math works out
         (reverse (rest hiddens)))))))


;;;;;; One-hot-embeddin init ;;;;;
(def embed-size config/embed-size)
(def guidance-token-size config/guidance-token-size)
(def extra-size-image-embed config/extra-size-image-embed)
(def zero-array (into [] (replicate embed-size 0)))
(def one-hot-vec (assoc zero-array 0 1) )
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


(defn sample-seq
  "Samples the network by considering the grammer of the DSL
   and returns num_samples samples of len max_seq_len.
  Outputs: sample, hidden
  - samples: num_samples x max_seq_lenght (a sample sequenced in each row)"
  [cell hidden->loggits inputs max_seq_len num_samples start_symbol end_symbol]
  (let [factory (-> cell cell-model model/tensor-factory)
        out-dim (output-dim cell)
        hidden (init-hidden cell)
        input (first inputs)
        in-dim (input-dim cell)]
    ; We check until we hit the stop symbol or it passes the maximum length of seq
    (loop [input input hiddens (list (init-hidden cell)) samples '() cnt 0 previous-samples [-1 -1 -1 -1 -1]]
      (if (and (< cnt max_seq_len) (< (last previous-samples) (- embed-size guidance-token-size)))
        (let [last-hidden (first hiddens)
              hidden (add-input cell input last-hidden)
              sample (cond
                (= cnt 0) one-hot-vec
               
                (= (last previous-samples) 0)
                (no/multinomial-dist (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value)) (assoc zero-array 1 1 2 1 3 1 end_symbol 1) ))

                ;"sampled shape entities-> sample from range 5 22"
                (< (last previous-samples) 4)
                (no/multinomial-dist
                  (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value))
                        (apply assoc zero-array (into [] (flatten (apply map vector [(into [] (range 6 22))
                                                                                     (into [] (replicate 16 1))])))) ))

                ;"it is in the process of selecting the coordinates for Rectangles and lines"
                ; just check for the first samples
                (and (> (last previous-samples) 5) (< (last previous-samples) 22)
                     (not (and (> (second previous-samples) 5)
                               (< (second previous-samples) 22)))
                               (not (= 3 (second previous-samples))))

                (no/multinomial-dist
                  (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                       :value))
                        (apply assoc zero-array (into [] (flatten (apply map vector [(into [] (range 6 22))
                                                                                     (into [] (replicate 16 1))])))) ))


                ;"it is in the process of selecting the coordinates for circles"
                (and (> (last previous-samples) 5) (< (last previous-samples) 22)
                     (or (= 3 (get previous-samples 3))
                         (= 3 (get previous-samples 2))))
                (no/multinomial-dist
                  (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                       :value))
                        (apply assoc zero-array (into [] (flatten (apply map vector [(into [] (range 6 22))
                                                                                     (into [] (replicate 16 1))])))) ))
                ;"it has already exhausted the arguments and time to sample from shape entities
                ; (Rectangles and Circles) or stop/end symbols"
                (and (> (last previous-samples) 5) (< (last previous-samples) 22)
                     (or (and (> (second previous-samples) 5) (< (second previous-samples) 22))
                         (= (second previous-samples) 3))
                     (not (= 1 (first previous-samples))))
                (no/multinomial-dist (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value)) (assoc zero-array 1 1 2 1 3 1 end_symbol 1) ))

                ;"It is time for the sampling from true or false (4-5) inclusive"
                (and (> (last previous-samples) 5) (< (last previous-samples) 22)
                     (> (second previous-samples) 5) (< (second previous-samples) 22)
                     (= 1 (first previous-samples)))
                (no/multinomial-dist (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value)) (assoc zero-array 4 1 5 1) ))

                ;"it has exhausted all the false-true args and time to sample shape entity again or stop/end symbol"
                (and (> (last previous-samples) 3) (< (last previous-samples) 6)
                     (> (get previous-samples 3) 3) (< (get previous-samples 3) 6))
                (no/multinomial-dist (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value)) (assoc zero-array 1 1 2 1 3 1 end_symbol 1) ))

                ;"it has sampled one true or false, it is time do this one more time (4-5) inclusive"
                (and (> (last previous-samples) 3) (< (last previous-samples) 6)
                     (not (and (> (get previous-samples 3) 3) (< (get previous-samples 3) 6))))
                (no/multinomial-dist (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value)) (assoc zero-array 4 1 5 1) ))

                ;"it has sampled the stop sign and it is time to sample the sketch guidance signals"
                (= (last previous-samples) (- embed-size guidance-token-size 1))
                (no/multinomial-dist (mapv * (into [] (-> (module/graph hidden->loggits (first hidden))
                                                          :value)) (apply assoc zero-array
                                                                          (into [] (flatten
                                                                                     (apply map vector
                                                                                            [(into [] (range (- embed-size guidance-token-size) embed-size))
                                                                                             (into [] (replicate guidance-token-size 1))])))) ))

                )
                current  (.indexOf sample 1)
                past-samples (if (< current 4)
                               [-1 -1 -1 -1 current]
                               [(get previous-samples 1)
                                (get previous-samples 2)
                                (get previous-samples 3)
                                (get previous-samples 4)
                                current])
                ]
          (recur (node/const sample) (cons hidden hiddens) (cons (.indexOf sample 1) samples)
                 (+ cnt 1) past-samples))
        (reverse samples)))))
