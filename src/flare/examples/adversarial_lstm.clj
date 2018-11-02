(ns flare.examples.adversarial-lstm
  (:require [clojure.pprint :as pp]
            [opennlp.nlp :as nlp])
  )
(def vocabulary_size 1000)
(def unknown_token "UNKNOWN_TOKEN")
(def sentence_start_token "SENTENCE_START ")
(def sentence_end_token " SENTENCE_END")

(def tokenize (nlp/make-tokenizer "data/en-token.bin"))
(def get-sentences (nlp/make-sentence-detector "data/en-sent.bin"))


;;delimit , tokenize and counts the words frequency
(defn word-freq-delimited [s]
  (->> s
       clojure.string/lower-case
       get-sentences
       (map #(str sentence_start_token % sentence_end_token))
       (mapcat tokenize)
       frequencies
       )
  )

;; selects the n most frequent words and index them
(defn words-index [n items]
  (->> items
       word-freq-delimited
       (sort-by val)
       reverse
       (take n)
       (map first)
       (map-indexed (fn [idx itm] [itm idx]))
       (into {}))
  )


(def sentences (slurp "data/sentiment-sentences-train.txt"))
(def words-indexed (words-index vocabulary_size sentences))

; (doseq [x words-indexed]
; (spit "data/words-to-one-hot-vectors.txt" (prn-str (str (first x) " " (apply pr-str (assoc (vec (replicate vocabulary_size 0)) (second x) 1)))) :append true))

;(pp/pprint (tokenize "Mr. Smith gave a car to his son on Friday"))
;(into {} (map-indexed (fn [idx itm] [itm idx]) (into #{} (mapcat tokenize (get-sentences
; "First sentence. Second sentence?Here is another one. And so on and so forth - you get the idea...")))))
; (into #{} (mapcat tokenize (get-sentences
; "First sentence. Second sentence?Here is another one. And so on and so forth - you get the idea...")))
; ((map tokenize (get-sentences
;                 "First sentence. Second sentence?Here is another one. And so on and so forth - you get the idea...")))
;(defn word-to-index [sentences]
; (into {} (map-indexed (fn [idx itm] [itm idx]) (into #{} (mapcat tokenize (get-sentences sentences))) ) ))
; (frequencies (mapcat tokenize (get-sentences
; "First sentence. Second sentence?Here is another one. And so on and so forth - you get the idea...")))


