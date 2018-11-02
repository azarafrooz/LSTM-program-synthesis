(ns adversarial-sketch.core
  (:gen-class)
  (:require [quil.core :as q :include-macros true]
            [quil.middleware :as m]
            [flare.examples.generator :as g]
            [tempfile.core :refer :all]
            [clojure.java.shell :refer [sh]]
            [clojure.java.io :as io]
            [clojure.string :as string])
  (:import sketch.compiler.main.seq.SequentialSketchMain
           [java.io ByteArrayOutputStream PrintStream]))


; https://stackoverflow.com/questions/37773329/capturing-system-out-in-clojure
(defmacro with-system-out-str [& body]
  `(let [out-buffer# (ByteArrayOutputStream.)
         original-out# System/out
         tmp-out# (PrintStream. out-buffer# true "UTF-8")]
     (try
       (System/setOut tmp-out#)
       ~@body
       (finally
         (System/setOut original-out#)))
     (.toString out-buffer# "UTF-8")))
;This is to run the lstm flare
;(g/-main)
;
;; common.skh needs to be copied in the same temp directory. A helper function.
(defn copy-file [source-path dest-path]
  (io/copy (io/file source-path) (io/file dest-path)))

;;;eh there is no logarithm2 in Math clojure
(defn log2 [n]
  (/ (Math/log n) (Math/log 2)))


;;extraxts the main body,cost from the sketch output
(defn get-body-sketchoutput [sketch-output user-prior]
  (if (or (string/includes? sketch-output "Sketch front-end timed out")
          (string/includes? sketch-output "The sketch cannot be resolved")
          (and user-prior
               (not (string/includes? sketch-output "INSIDE minimizeHoleValue"))))
    nil
    (let [cost (if user-prior
                 (let [programSize (string/split-lines sketch-output)]
                   (Integer.
                     (get (re-find (re-matcher #".*=([0-9]+)," (last sketch-output))) 1)))

                 -1)
          body (loop [output (string/split-lines sketch-output)
                      body nil]
                 (if (or (string/includes? (first output) "minimize ") (nil? (next output)))
                   body
                   (recur (next output)
                          (cond
                            (and (string/includes? (first output) "void render ") (= body nil)) [(first output)]
                            (not= body nil ) (conj body (first output))
                            :else body))))]
      {:cost cost :body (clojure.string/join "\n" body)})))



; The entry code for sythesizing part

(defn parse-lstm-ouput [parse-lines]
  (let [parts [] xValidation []
        yValidation [] solid [] arrows [] hasLines false hasRectangles false hasCircles false
        syntaxDic {:parts parts :xValidation xValidation :yValidation yValidation
                   :solid solid :arrows arrows
                   :hasLines      hasLines
                   :hasRectangles hasRectangles
                   :hasCircles    hasCircles}]
    (loop [parse-lines parse-lines syntaxDic syntaxDic]
      (if (empty? parse-lines)
        syntaxDic
        (case (first parse-lines)
          1 (recur (into [] (next (next (next (next (next (next (next parse-lines))))))))
                   (assoc syntaxDic :parts (conj (:parts syntaxDic) (format "_l(%d,%d,%d,%d,%d,%d)"
                                                                            (- (get parse-lines 1) 6)
                                                                            (- (get parse-lines 2) 6)
                                                                            (- (get parse-lines 3) 6)
                                                                            (- (get parse-lines 4) 6)
                                                                            (if (= 4 (get parse-lines 5)) 1 0)
                                                                            (if (= 4 (get parse-lines 6)) 1 0)
                                                                            ))
                                    :xValidation (conj (:xValidation syntaxDic) (- (get parse-lines 1) 6) (- (get parse-lines 3) 6))
                                    :yValidation (conj (:yValidation syntaxDic) (- (get parse-lines 2) 6) (- (get parse-lines 4) 6))
                                    :hasLines true
                                    :arrows (conj (:arrows syntaxDic) (if (= 4 (get parse-lines 5)) 1 0))
                                    :solid (conj (:arrows syntaxDic) (if (= 4 (get parse-lines 6)) 1 0))))
          2 (recur (into [] (next (next (next (next (next parse-lines))))))
                   (assoc syntaxDic
                     :parts (conj (:parts syntaxDic)
                                  (format "_r(%d,%d,%d,%d)"
                                          (- (get parse-lines 1) 6)
                                          (- (get parse-lines 2) 6)
                                          (- (get parse-lines 3) 6)
                                          (- (get parse-lines 4) 6)))
                     :xValidation
                     (conj (:xValidation syntaxDic) (- (get parse-lines 1) 6) (- (get parse-lines 3) 6))
                     :yValidation
                     (conj (:yValidation syntaxDic) (- (get parse-lines 2) 6) (- (get parse-lines 4) 6)
                           )
                     :hasRectangles true))
          3 (recur (into [] (next (next (next (next parse-lines)))))
                   (assoc syntaxDic :parts (conj (:parts syntaxDic) (format "_c(%d,%d)" (- (get parse-lines 1) 6)
                                                                            (- (get parse-lines 2) 6)))
                                    :xValidation (conj (:xValidation syntaxDic) (- (get parse-lines 1) 6))
                                    :yValidation (conj (:yValidation syntaxDic) (- (get parse-lines 2) 6))
                                    :hasCircles true))

          "The DSL syntax is not supported: core.clj")))))


(defn synthesize-program [parse-lines]
  "Consider the sketch program itself as stream of string tokens augmented with the LSTM predictions."
  (let [syntaxDic (parse-lstm-ouput parse-lines)
        xCoefficients ""
        yCoefficients ""
        usedReflections []
        usedLoops []
        noDiagonals false
        x1 16                                              ;max x distance
        y1 16                                               ;max y distance
        biggestNumber -1 CPUs 8 timeout 10 usePrior false maximumDepth 3 canLoop true canReflect false
        smallestPossibleLoss (+ 1 (* 3 (count parse-lines)))
        upperBoundOnLoss (format "--bnd-mbits %d" (min 5 (int (Math/ceil (log2 (+ 1 smallestPossibleLoss))))))
      
        coefficientGenerator1 (clojure.string/join " | " (map str xCoefficients))
        coefficientGenerator2 (clojure.string/join " | " (map str yCoefficients))

        ;;code has issue in the following line, I fixed by putting dummy values!
        coefficientValidator1 "c == 1 || c==2 || c==3 || c==4 || c==5 || c==6 || c==7" ;;displacement in the python codes are not yet implemented
        coefficientValidator2 "c == 1 || c==2 || c==3 || c==4 || c==5 || c==6 || c==7" ;displacement in the python codes are not yet implemented
        xValidation  (clojure.string/join " || " (map #(format "x == %d" %) (set (:xValidation syntaxDic))))
        yValidation  (clojure.string/join " || " (map #(format "x == %d" %) (set (:yValidation syntaxDic))))
        haveThisReflectionAlready (str
                                    (clojure.string/join " || "
                                                         (conj (into [] (map #(format "(xr == %d && yr == %d)" % %) usedReflections))
                                                               "0")))
        alreadyProvidedBounds (loop [bound usedLoops returnValue (str (:intercept bound)) alreadyProvidedBounds ""]
                                (if (empty? bound)
                                  alreadyProvidedBounds
                                  (recur
                                    (next usedLoops)
                                    ((if (and (not (:coefficient bound)) (not (= (:coefficient bound) 0)))
                                       (conj returnValue (format " + %s*environment[%d]" (:coefficient bound) (:variable bound)))
                                       (returnValue)
                                       ))
                                    (conj alreadyProvidedBounds (format " if (n == %d && ??) { already_have_this_loop = 1; return %s; }"
                                                                        (:depth bound) returnValue)))))

        source-code (format "
pragma options \"--fe-timeout %d --bnd-unroll-amnt 4 --bnd-arr1d-size 2 --bnd-arr-size 2 --bnd-int-range %d %s\";

%s
#define MAXIMUMDEPTH %d
#define CANLOOP %d
#define CANREFLECT %d
#define ALREADYPROVIDEDBOUNDS %s
#define HAVETHISREFLECTIONALREADY %s
#define XCOEFFICIENTS %s
#define YCOEFFICIENTS %s
#define PROVIDEDXCOEFFICIENTS %d
#define PROVIDEDYCOEFFICIENTS %d
#define XVALIDATION ( %s )
#define YVALIDATION ( %s )
#define COEFFICIENTVALIDATOR1 ( %s )
#define COEFFICIENTVALIDATOR2 ( %s )
#define MAXIMUMLOOPITERATIONS 4
#define MAXIMUMXCOORDINATE %d
#define MAXIMUMYCOORDINATE %d
#define HASCIRCLES %d
#define HASRECTANGLES %d
#define HASLINES %d
#define HASSOLID %d
#define HASDASHED %d
#define HASARROW %d
#define HASNOARROW %d
#define NODIAGONALS %d

#define COSTUPPERBOUND %d

%s

bit renderSpecification(SHAPEVARIABLES) {
                                         assume shapeIdentity == CIRCLE || shapeIdentity == LINE || shapeIdentity == RECTANGLE;
                                         if (!HASCIRCLES) assume shapeIdentity != CIRCLE;
                                         if (!HASRECTANGLES) assume shapeIdentity != RECTANGLE;
                                         if (!HASLINES) assume shapeIdentity != LINE;
                                         return %s;
                                         }

" timeout biggestNumber upperBoundOnLoss (if usePrior "#define USEPRIOR" "") maximumDepth (if canLoop 1 0) (if canReflect 1 0)
                            alreadyProvidedBounds
                            haveThisReflectionAlready
                            coefficientGenerator1 coefficientGenerator2
                            (count xCoefficients)
                            (count yCoefficients)
                            xValidation yValidation
                            coefficientValidator1 coefficientValidator2
                            x1 y1
                            (if (:hasCircles syntaxDic) 1 0)
                            (if (:hasRectangles syntaxDic) 1 0)
                            (if (:hasLines syntaxDic) 1 0)
                            (if (some  #(= 1 %) (:solid syntaxDic)) 1 0)
                            (if (some  #(= 0 %) (:solid syntaxDic)) 1 0)
                            (if (some  #(= 1 %) (:arrows syntaxDic)) 1 0)
                            (if (some #(= 0 %) (:arrows syntaxDic)) 1 0)
                            (if noDiagonals 1 0)
                            (count parse-lines)
                            ; ;hack for import "common.skh"
                            ;commons.shk should be its path but copying it
                            ; on the path giving parsing errors
                            (slurp "common.skh")
                            (clojure.string/join " || " (:parts syntaxDic)))]

    (with-tempfile [t (tempfile source-code)]
                   (let [exists (.exists t)
                         fname (.getAbsolutePath t)
                         fdir ()
                         content (sh "cat" fname)]
                     (get-body-sketchoutput (with-system-out-str (SequentialSketchMain/main (into-array String [fname ]) ) ) usePrior)))))
