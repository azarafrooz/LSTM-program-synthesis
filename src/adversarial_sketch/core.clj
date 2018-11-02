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
  ;;usedLoops: [{depth, coefficient, variable, intercept}]
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
        ;coefficientGenerator1 ""
        ;coefficientGenerator2 ""
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
                     ;(sh "java" "-cp" "/Users/mahdiazarafrooz/Desktop/Human/adversarial-sketch/sketch-1.7.5-noarch.jar" "sketch.compiler.main.seq.SequentialSketchMain" fname)
                     ;(SequentialSketchMain/main (into-array String [fname ]) )
                     ;with-out-str to redirect stdout to string
                     ;(print (with-out-str (SequentialSketchMain/main (into-array String [fname ]) ) ))
                     (get-body-sketchoutput (with-system-out-str (SequentialSketchMain/main (into-array String [fname ]) ) ) usePrior)))))
;(get-body-sketchoutput (:out (SequentialSketchMain/main (into-array String [fname ]) ) ) usePrior)
;(SequentialSketchMain/main (into-array String ["--fe-output-code" fname]) )




;<< shell way to run the Sketch solver >>
;(use '[clojure.java.shell :only [sh]])
;(def ans (sh "java" "-cp" "/Users/mahdiazarafrooz/Desktop/Human/adversarial-sketch/sketch-1.7.4/sketch-frontend/sketch-1.7.4-noarch.jar" "sketch.compiler.main.seq.SequentialSketchMain" "isolateRightmost.sk"))
;(:out ans)

;<<non-shellway to run the Sketch Solver>>
;(import 'sketch.compiler.main.seq.SequentialSketchMain)
;(SequentialSketchMain/main (into-array String ["isolateRightmost.sk"]))
;(SequentialSketchMain/main (into-array String ["--fe-output-code" "isolateRightmost.sk"]))


;Temporary files is essential here since to interact with sketch program
; we need a file-based API.
; Using createTempFile is important to ensure that temporary files are
; placed in an appropriate location on the filesystem, which can differ
; based on the operating system being used.


;(defn -main
;  ;  "I don't do a whole lot ... yet."
;  ;  (g/-main))
;  [& args]
;  ;(print (synthesize-program [2 6 7 8 9 2 7 8 9 10]))
;  ;(def output-example "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpzqJj8W.sk:210\"; //Assume at tmpzqJj8W.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpzqJj8W.sk:212\"; //Assume at tmpzqJj8W.sk:212\n  assume (!(dashed)): \"Assume at tmpzqJj8W.sk:216\"; //Assume at tmpzqJj8W.sk:216\n  assume (!(arrow)): \"Assume at tmpzqJj8W.sk:217\"; //Assume at tmpzqJj8W.sk:217\n  int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)\n        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/\n        {\n          dummyStartBoundary();\n          bit _pac_sc_s26 = _pac_sc_s15_s17_0;\n          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n            _pac_sc_s26 = _pac_sc_s26_s30;\n          }\n          _pac_sc_s15_s17_0 = _pac_sc_s26;\n          dummyEndBoundary();\n        }\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39_1 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);\n          int y_s43_1 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1))")
;  (parser/parse-sketch-output adversarial-sketch.sketch-parser/output-example {} {} 0)
;  )

;(defn -main
;  ;  "I don't do a whole lot ... yet."
;  ;  (g/-main))
;  [& args]
;  (print (synthesize-program [2 6 7 8 9 2 7 8 9 10]))
;  )