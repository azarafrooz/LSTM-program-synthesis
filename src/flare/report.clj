(ns flare.report
  (:refer-clojure :exclude [concat])
  (:require [flare.model :as model]
            [clojure.spec.alpha :as s]))

(defprotocol Reporter
  "For tracking behavior of model across examples"
  (update! [this info]
    "update with `info` after an 'event'")
  (clear! [this]
    "clear internal state")
  (gen [this]
    "make a map report of observations"))

(defn accuracy
  "Creare report entry with `key`-accuracy key (e.g, :train, :test) and the 
   value is the accuracy from using `predict-fn` over the data generated 
   by `get-data` which is assumed to return a `[x label]` pair. 
   The `label` is assumed to be an integer type number and is cast as 
  long to compare to  `(predict-fn data)`"
  [key get-data predict-fn]
  (reify Reporter
    (update! [this info])
    (clear! [this])
    (gen [this]
      (let [[^long num-correct ^long total]
            (->> (get-data)
                 (map (fn [[x label]]
                        [(predict-fn x) label]))
                 (filter first)
                 (map (fn [[prediction label]]
                        (= (long label) (long prediction))))
                 (reduce (fn [[^long num-correct ^long total] correct?]
                           [(if correct? (inc num-correct) num-correct)
                            (inc total)])
                         [0 0]))]
        {(keyword (str (name key) "-accuracy"))
         {:acc (/ (double num-correct) total) :n total}}))))

(defn callback
  "reporter which just uses callback `f`"
  [f]
  (reify Reporter
    (update! [this info])
    (clear! [this])
    (gen [this] (f))))

(defn avg-loss []
  (let [sum (atom 0.0)
        n (atom 0)]
    (reify
      Reporter
      (update! [this info]
        (swap! sum + (:batch-loss info))
        (swap! n + (count (:batch info))))
      (gen [this]
        {:avg-loss {:avg (/ @sum @n) :n @n}})
      (clear! [this]
        (reset! sum 0.0)
        (reset! n 0)))))

(defn grad-size []
  (let [sum-l2-norm (atom 0.0)
        sum-max-l1-norm (atom 0.0)
        n (atom 0)]
    (reify
      Reporter
      (update! [this info]
        (let [factory (model/tensor-factory (:model info))
              grads (mapcat (fn [[_ x]] (flatten (seq (:grad x))))
                            (:model info))
              l2-norm (Math/sqrt (reduce (fn [res x] (+ res (* x x))) 0.0 grads))
              max-l1-norm (apply max (map (fn [x] (Math/abs (double x))) grads))]
          (swap! sum-l2-norm + l2-norm)
          (swap! sum-max-l1-norm + max-l1-norm)
          (swap! n inc)))
      (gen [this]
        {:l2-norm {:avg (/ @sum-l2-norm @n) :n @n}
         :max-l1-norm {:avg (/ @sum-max-l1-norm @n) :n @n}})
      (clear! [this]
        (reset! sum-l2-norm 0.0)
        (reset! sum-max-l1-norm 0.0)
        (reset! n 0)))))

(defn every [freq r]
  (let [n (atom 0)]
    (reify
      Reporter
      (update! [this info]
        (update! r info))
      (gen [this]
        (swap! n inc)
        (when (zero? (mod @n freq))
          (gen r)))
      (clear! [this]
        (reset! n 0)
        (clear! r)))))

