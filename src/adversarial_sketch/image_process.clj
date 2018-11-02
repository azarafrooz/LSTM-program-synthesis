(ns adversarial-sketch.image-process
  (:require [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [think.image.image-util :as image-util]
            [mikera.image.filters :as filt]
            )
  (:use [mikera.image.core]))

(def max-image-rotation-degrees 32)

(defn img-aug-pipeline
  [img]
  (-> img
      (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                           max-image-rotation-degrees)
                        false)
      (image-aug/inject-noise (* 0.25 (rand)))))

(defn png->observation
  "Create an observation from input.  "
  [datatype augment? image-file]
  (let [img (load-image image-file)] ;;image->patch always returns [r-data g-data g-data]
    ;;since we know these are grayscale *and* we setup the
    ;;network for 1 channel we just take r-data
    (first (patch/image->patch (if augment?
                                 (img-aug-pipeline img)
                                 img)
                               (image-util/image->rect img) datatype))))


(defn transpose [m]
  (apply mapv vector m))

(defn regex-file-seq
  "Lazily filter a directory based on a regex."
  [re dir]
  (filter #(re-find re (.getPath %)) (file-seq dir)))

(defn get-data-matrix [path-to-files]
  (->> (regex-file-seq #".png" (clojure.java.io/file path-to-files))
       (map #(png->observation :float false %))))