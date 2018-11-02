(ns adversarial-sketch.graphical-language
  (:require [quil.core :as q :include-macros true]
            [quil.middleware :as m]
            [adversarial-sketch.sketch-parser :as sketch-parser]
            [clojure.string :as string]
            [instaparse.core :as insta]
            [adversarial-sketch.core :as synthesizer]))


(defn rectangular-draw
  [x1 y1 x2 y2]
  (q/translate 1 3)
  (doseq [t (range 0 10 1)]
    (q/rect (* 16 x1) (* 16 y1) (* 16 (- x2 x1)) (* 16 (- y2 y1)))
    (if (> t 5)
      (q/line (* 16 x1) (* 16 y1) (* 16 x2) (* 16 y2))))
  (q/translate 1 3))


(defn ellipse-draw [x1 y1 r]
  (q/ellipse (* 16 x1) (* 16 y1) (* 32 r) (* 32 r)))


(defn line-draw [x1 y1 x2 y2]
  ; make background white
  (q/line (* 16 x1) (* 16 y1) (* 16 x2) (* 16 y2)))



(defn draw []
  ; make background white
  (q/background 255)

  ; move origin point to centre of the sketch
  ; by default origin is in the left top corner
  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]
                      ; parameter t goes 0, 0.01, 0.02, ..., 99.99, 100
                      (doseq [t (range 0 100 0.01)]
                        ; draw a point with x = t * sin(t) and y = t * cos(t)
                        (line-draw (* t (q/sin t))
                                   (* t (q/cos t))  (* t (q/sin t))  (* t (q/sin t))))))





(defn draw-loops3 []
  ; make background white
  (q/background 255)

  ; move origin point to centre of the sketch
  ; by default origin is in the left top corner
  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]
                      ; parameter t goes 0, 0.01, 0.02, ..., 99.99, 100
                      (do (rectangular-draw 5 5 3 4))
                      (do (line-draw 1 2 1 4))
                      (do (ellipse-draw 0 10 4))
                      (doseq [i (range 0 0 2)]
                        (rectangular-draw (* i (q/sin i))
                                          (* i (q/cos i))  (* i (q/sin i))  (* i (q/sin i)))
                        (line-draw (* i (q/sin i))
                                   (* i (q/cos i))  (* i (q/sin i))  (* i (q/sin i)))
                        (ellipse-draw (* i (q/sin i))
                                      (* i (q/cos i)) 1)
                        (doseq [j (range 0 0 20)]
                          (rectangular-draw (* i (q/sin j))
                                            (* i (q/cos i))  (* j (q/sin j))  (* i (q/sin i)))
                          (line-draw (* i (q/sin i))
                                     (* i (q/cos i))  (* i (q/sin i))  (* i (q/sin i)))
                          (ellipse-draw (* i (q/sin i))
                                        (* j (q/cos j)) 1)))))



;; run sketch
;(q/defsketch trigonometry
;             :size [300 300]
;             :draw draw-new)





"(defn draw-loops2 []
; make background white
  (q/background 255)

  ; move origin point to centre of the sketch
  ; by default origin is in the left top corner
  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]
                      ; parameter t goes 0, 0.01, 0.02, ..., 99.99, 100
                      (do (rectangular-draw 5 5 3 4))
                      (do (line-draw 1 2 1 4))
                      (do (ellipse-draw 0 10 4))
                      (doseq [i (range 0 10 2)]
                        (rectangular-draw (* i (q/sin i))
                                          (* i (q/cos i))  (* i (q/sin i))  (* i (q/sin i)))
                        (line-draw (* i (q/sin i))
                                   (* i (q/cos i))  (* i (q/sin i))  (* i (q/sin i)))
                        (ellipse-draw (* i (q/sin i))
                                      (* i (q/cos i)) 1)
                        (doseq [j (range 0 100 20)]
                          (rectangular-draw (* i (q/sin j))
                                     (* i (q/cos i))  (* j (q/sin j))  (* i (q/sin i)))
                          (line-draw (* i (q/sin i))
                                     (* i (q/cos i))  (* i (q/sin i))  (* i (q/sin i)))
                          (ellipse-draw (* i (q/sin i))
                                        (* j (q/cos j)) 1)))))
"




;(defn draw-test []
;; make background white
;  (q/background 255)
;
;  ; move origin point to centre of the sketch
;  ; by default origin is in the left top corner
;  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]
;                      ; parameter t goes 0, 0.01, 0.02, ..., 99.99, 100
;                      (doseq [i (range 0 3 1)]
;                        (doseq [j (range 0 3 1)]
;                          (ellipse-draw (+ (* -3 j) 7) (+ (* -3 i) 7) 1)
;                          (line-draw (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9) (+ (* -3 i) 7))
;                          (line-draw (+ (* -3 i) 7) (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9))))))
;


(defn draw-test2 []
  (q/background 255)\n         (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)])(doseq [i (range 0 3 1)](doseq [j (range 0 3 1)](ellipse-draw (+ (* -3 j) 7) (+ (* -3 i) 7) 1)(line-draw (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9) (+ (* -3 i) 7))(line-draw (+ (* -3 i) 7) (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9)))))


(defn draw-test3 []
  (doseq [i (range 0 3 1)] (doseq [j (range 0 3 1)] (ellipse-draw (+ (* -3 j) 7) (+ (* -3 i) 7) 1) (line-draw (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9) (+ (* -3 i) 7)) (line-draw (+ (* -3 i) 7) (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9)))))

(defn draw-test4 []
  (ellipse-draw (+ (* 0 0) 4) (+ (* 0 0) 14) 1) )



(defn setup []
  (q/background 255)
  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]))


;; Parser and compilers

(def compiler-parser-bnf
  (insta/parser
    (clojure.java.io/resource "graphic_compiler_parser.bnf")))

(defn graphic-compiler-parser
  "trims, compiles to the graphic-dsl via insta lib"
  [sketch-ast]
  (->> (string/replace sketch-ast #"[\[|\]|\"|\{|\}|,]|:|="  "")
       compiler-parser-bnf))

(defn visualize-ast
  "visulize the ast"
  [sketch-ast]
  (insta/visualize (graphic-compiler-parser sketch-ast)) )


(defn graphic-compiler-transformer [sketch-ast]
  (->> (graphic-compiler-parser sketch-ast)
       (insta/transform
         {:loops
          (fn [loop-var start-indx variable loop-offset ]
            (let [loop-var  (if (= loop-var "nil") 0 loop-var)
                  start-indx (Integer. start-indx)
                  loop-offset (Integer. loop-offset)]
              `(doseq [~loop-var (range ~start-indx ~loop-offset 1)])))
          :circle
          (fn [x-factor x-variable x-offset y-factor y-variable y-offset ] (let [x-variable (if (= x-variable "nil") 0 x-variable)
                                                                                 y-variable  (if (= y-variable "nil") 0 y-variable)
                                                                                 x-offset (Integer. x-offset)
                                                                                 y-offset (Integer. y-offset)
                                                                                 x-factor (Integer. x-factor)
                                                                                 y-factor (Integer. y-factor)]
                                                                             `(ellipse-draw (+ (* ~x-factor ~x-variable) ~x-offset)
                                                                                            (+ (* ~y-factor ~y-variable) ~y-offset)
                                                                                            1)))

          :line
          (fn [x1-factor x1-variable x1-offset y1-factor y1-variable y1-offset
               x2-factor x2-variable x2-offset y2-factor y2-variable y2-offset
               arrow solid]
            (let [x1-variable (if (= x1-variable "nil") 0 x1-variable)
                  y1-variable  (if (= y1-variable "nil") 0 y1-variable)
                  x1-offset (Integer. x1-offset)
                  y1-offset (Integer. y1-offset)
                  x1-factor (Integer. x1-factor)
                  y1-factor (Integer. y1-factor)
                  x2-variable (if (= x2-variable "nil") 0 x2-variable)
                  y2-variable  (if (= y2-variable "nil") 0 y2-variable)
                  x2-offset (Integer. x2-offset)
                  y2-offset (Integer. y2-offset)
                  x2-factor (Integer. x2-factor)
                  y2-factor (Integer. y2-factor)]
              `(line-draw (+ (* ~x1-factor ~x1-variable) ~x1-offset) (+ (* ~y1-factor ~y1-variable) ~y1-offset)
                          (+ (* ~x2-factor ~x2-variable) ~x2-offset) (+ (* ~y2-factor ~y2-variable) ~y2-offset))))
          :rectangle
          (fn [x1-factor x1-variable x1-offset y1-factor y1-variable y1-offset
               x2-factor x2-variable x2-offset y2-factor y2-variable y2-offset]
            (let [x1-variable (if (= x1-variable "nil") 0  x1-variable)
                  y1-variable  (if (= y1-variable "nil") 0 y1-variable)
                  x1-offset (Integer. x1-offset)
                  y1-offset (Integer. y1-offset)
                  x1-factor (Integer. x1-factor)
                  y1-factor (Integer. y1-factor)
                  x2-variable (if (= x2-variable "nil") 0 x2-variable)
                  y2-variable  (if (= y2-variable "nil") 0 y2-variable)
                  x2-offset (Integer. x2-offset)
                  y2-offset (Integer. y2-offset)
                  x2-factor (Integer. x2-factor)
                  y2-factor (Integer. y2-factor)]
              `(rectangular-draw (+ (* ~x1-factor ~x1-variable) ~x1-offset)
                                 (+ (* ~y1-factor ~y1-variable) ~y1-offset)
                                 (+ (* ~x2-factor ~x2-variable) ~x2-offset)
                                 (+ (* ~y2-factor ~y2-variable) ~y2-offset))))})))



;(defn eval-graphic-draw [sketch-ast]
;  (-> (cons `(\( defn draw-magic  [])  (graphic-compiler-transformer sketch-ast))
;      print))


;(defn eval-graphic-draw [sketch-ast]
;  (-> `( \( defn draw-magic  [] ~@(graphic-compiler-transformer sketch-ast) \)\))
;      print))





;(defmacro eval-graphic-draw [sketch-ast]
;  `(defn draw-magic [] ~@(graphic-compiler-transformer sketch-ast)))

;;;;;;test stufff
;(declare draw-magic)
;(defn run-draw [lstm-out]
;  (let [sketch-ast  (-> (:body (synthesizer/synthesize-program lstm-out))
;                          string/split-lines
;                          (sketch-parser/parse-sketch-output {} {} 0)
;                          :commands)]
;      (q/defsketch trigonometry
;                       :title "compiled codes"
;                       :setup setup
;                       :size [300 300]
;                       :draw (fn drawfn []
;                               (doseq [i (range 0 3 1)]
;                                 (doseq [j (range 0 3 1)]
;                                   (ellipse-draw (+ (* -3 j) 7) (+ (* -3 i) 7) 1)
;                                   (line-draw (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9) (+ (* -3 i) 7))
;                                   (line-draw (+ (* -3 i) 7) (+ (* -3 j) 8) (+ (* -3 i) 7) (+ (* -3 j) 9))))))))

;;;;;;;;;;;;; Macro and elegebanr way
;;(defmacro compiler-macro [sketch-ast]
;;  (-> `(graphic-compiler-transformer ~@sketch-ast)
;;      first))
;
;(defn run-draw [lstm-out]
;  (let [sketchast  (-> (:body (synthesizer/synthesize-program lstm-out))
;                        string/split-lines
;                        (sketch-parser/parse-sketch-output {} {} 0)
;                        :commands)]
;    (println sketchast)
;    (println (macroexpand-1 `(graphic-compiler-transformer ~@sketchast)))
;    (q/defsketch trigonometry
;                 :title "compiled codes"
;                 :setup setup
;                 :size [300 300]
;                 :draw (fn drawfn []
;                         ;(eval (macroexpand-1 `(graphic-compiler-transformer ~@sketchast)))))))
;                          (graphic-compiler-transformer ~@sketchast)))))
;
;
;
;






;;;;;;it works totally (without a macro)...I would like to make it fancier with macro tho

;;(defmacro compiler-macro [sketch-ast]
;;  `(graphic-compiler-transformer ~sketch-ast))
;

(defn eval-transformer [sketch-ast]
  (graphic-compiler-transformer sketch-ast))

(defn eval-transformer-new [sketch-ast]
  (concat (first (eval-transformer sketch-ast))
          [(concat (second (eval-transformer sketch-ast))
                   (rest (rest (eval-transformer sketch-ast))))]))

(defn eval-transformer-general [sketch-ast]
    (loop [traversed-expressions (reverse (eval-transformer sketch-ast))
           ast ()
           loop-depth 0]

      (if (nil? (first traversed-expressions))
       (if (seq? (first ast))
         ast
         [ast])
        (cond
          (and
            (string/includes? (first traversed-expressions) "doseq")
            (or (nil? (second traversed-expressions))
                 (not (string/includes? (second traversed-expressions) "doseq")))
            (= loop-depth 0))
          (recur (next traversed-expressions) (concat (first traversed-expressions) ast) 0)
          (and
            (string/includes? (first traversed-expressions) "doseq")
            (and (not (nil? (second traversed-expressions)))
                 (string/includes? (second traversed-expressions) "doseq"))
            (= loop-depth 0))
          (recur (next traversed-expressions) [(concat (first traversed-expressions) ast)] 1)
          (and
            (string/includes? (first traversed-expressions) "doseq")
            (= loop-depth 1))
          (recur (next traversed-expressions) (concat (first traversed-expressions) ast) 0)
          :else (recur (next traversed-expressions) (cons (first traversed-expressions) ast) loop-depth)))))



(defn run-draw [lstm-out]
  (let [sketch-ast  (-> (:body (synthesizer/synthesize-program lstm-out))
                        (string/split #"[\n\|\|]")
                        (sketch-parser/parse-sketch-output {} {} 0)
                        :commands)]
    ;(println sketch-ast)
    ;(print (graphic-compiler-transformer sketch-ast))
    ;(print (graphic-compiler-transformer (string/join " " sketch-ast)))
    ;(println (macroexpand-1 `(graphic-compiler-transformer ~@sketch-ast)))
    (q/defsketch trigonometry
                 :title "compiled codes"
                 :setup setup
                 :size [300 300]
                 :draw (fn drawfn []
                         (doall (map eval-transformer sketch-ast))))))


(defn run-draw-ast [lstm-out]
  (let [sketch-ast  (-> (string/split lstm-out #"[\n\|\|]")
                        (sketch-parser/parse-sketch-output {} {} 0)
                        :commands)]
    ;(println sketch-ast)
    ;(print (graphic-compiler-transformer (string/join " " sketch-ast)))
    ;(print (type (eval-transformer sketch-ast)))
    ;(print (eval-transformer sketch-ast))
    ;(print (reduce cons (eval-transformer sketch-ast)))
    ;(print (apply concat (eval-transformer sketch-ast)))
    ;(print (concat (first (eval-transformer sketch-ast)) (second (eval-transformer sketch-ast))
    ;                       (rest (rest (eval-transformer sketch-ast)))))
    ;(print (concat (take 2 (eval-transformer sketch-ast)) (rest (rest (eval-transformer sketch-ast) ))))
    ;(print (reduce apply (eval-transformer sketch-ast)))
    ;(print (concat (second (eval-transformer sketch-ast)) (rest (rest (eval-transformer sketch-ast)))))
    ;(print (eval-transformer2 sketch-ast))
    ;(println (macroexpand-1 `(eval-transformer-new ~sketch-ast)))
    ;(println (eval-transformer-new sketch-ast))

    ;;(print (apply concat (first (eval-transformer sketch-ast)) (rest (eval-transformer sketch-ast))))
    ;(print (concat (concat (second (eval-transformer sketch-ast))
    ;                     (rest (rest (eval-transformer sketch-ast))))
    ;;             (first (eval-transformer sketch-ast))))
    ;(print (apply concat (rest (eval-transformer sketch-ast))))
    ;(print (rest (eval-transformer sketch-ast)))
    ;(print (map graphic-compiler-transformer sketch-ast))
    ;(println (macroexpand-1 `(graphic-compiler-transformer ~@sketch-ast)))
    (let [fns (eval-transformer-new sketch-ast)]
      (spit "generated_code.clj" (print-str fns))
      (print (slurp "generated_code.clj"))
      (q/defsketch trigonometry
                   :title "compiled codes"
                   :setup setup
                   :size [300 300]
                   :draw (fn drawfn []
                           (eval (read-string (slurp "generated_code.clj"))))))))


(defn run-draw-loopfree-ast [lstm-out]
  (let [sketch-ast  (-> (string/split lstm-out #"[\n\|\|]")
                        (sketch-parser/parse-sketch-output {} {} 0)
                        :commands)]
    (let [fns (eval-transformer sketch-ast)
          fns-string (print-str fns)]
      ;(spit "generated_code.clj" (subs fns-string 1 (- (count fns-string) 1))) ;remove extra paranthesis
      ;(type fns)
      (spit "generated_code.clj" fns-string) ;remove extra paranthesis
      (print (slurp "generated_code.clj"))
      (q/defsketch trigonometry
                   :title "compiled codes"
                   :setup setup
                   :size [300 300]
                   :draw (fn drawfn []
                           (doall (map eval (read-string (slurp "generated_code.clj"))))
                           )))
    ))


(defn run-draw-general-ast [lstm-out]
  (let [sketch-ast  (-> (string/split lstm-out #"[\n\|\|]")
                        (sketch-parser/parse-sketch-output {} {} 0)
                        :commands)]
    (let [fns (eval-transformer-general sketch-ast)
          fns-string (print-str fns)]
      ;(spit "generated_code.clj" (subs fns-string 1 (- (count fns-string) 1))) ;remove extra paranthesis
      ;(print (type (first fns)))
      (spit "generated_code.clj" fns-string) ;remove extra paranthesis
      ;(print (slurp "generated_code.clj"))
      (q/defsketch trigonometry
                   :title "compiled codes"
                   :setup setup
                   :size [300 300]
                   :draw (fn drawfn []
                           (doall (map eval (read-string (slurp "generated_code.clj"))))
                           )))
    ))

;;;;;;;;back to this when other stuff didn't work
;(defn eval-graphic-draw [sketch-ast]
;  `(defn draw-magic [] ~@(graphic-compiler-transformer sketch-ast)))
;
;
;(declare draw-magic)
;
;(defn run-draw [lstm-out]
;  (do (-> (:body (synthesizer/synthesize-program lstm-out))
;          string/split-lines
;          (sketch-parser/parse-sketch-output {} {} 0)
;          :commands
;          eval-graphic-draw))
;
;  (q/defsketch trigonometry
;               :title "compiled codes"
;               :setup setup
;               :size [300 300]
;               :draw draw-magic))

;;;;;;;;back to this when other stuff didn't work

;(defn run-draw [lstm-out]
;  (eval-graphic-draw lstm-out))
;

;(defn run-draw [lstm-out]
;  (->  (:body (synthesizer/synthesize-program lstm-out))
;           string/split-lines
;           (sketch-parser/parse-sketch-output {} {} 0)
;           :commands
;           eval-graphic-draw))
;(q/defsketch trigonometry
;             :title "compiled codes"
;             :setup setup
;             :size [300 300]
;             :draw graphic-compiler-transformer)


;(defn eval-graphic-draw [sketch-ast]
;  (->> (-> (graphic-compiler-transformer sketch-ast)
;           string/join
;           (str ")))"))
;       (str "(defn draw-magic []")
;       print))


;(->  (:body (synthesizer/synthesize-program [3 10 10 1]))
;     string/split-lines
;     (sketch-parser/parse-sketch-output {} {} 0)
;     :commands
;     eval-graphic-draw)
;(declare draw-magic)

;(defn run-draw [sketch-parser]
;          (do (->  (:body (synthesizer/synthesize-program [3 10 10 1]))
;                   string/split-lines
;                   (sketch-parser/parse-sketch-output {} {} 0)
;                   :commands
;                   eval-graphic-draw)))
;          (q/defsketch trigonometry
;                       :title "compiled codes"
;                       :setup setup
;                       :size [300 300]
;                       :draw draw-magic)))


;;(defn dsl-draw []
;  (q/defsketch trigonometry
;               :title "compiled codes"
;               :setup setup
;               :size [300 300]
;               :draw draw-magic))

;;; run sketch
;(q/defsketch trigonometry
;             :title "compiled codes"
;             :setup setup
;             :size [300 300]
;             :draw draw-magic)





;(defn draw2
;  "Example of a draw in quil"
;  []
;  ; make background white
;  (q/background 255)
;  ; move origin point to centre of the sketch
;  ; by default origin is in the left top corner
;  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]
;                      ; parameter t goes 0, 0.01, 0.02, ..., 99.99, 100
;                      (doseq [t (range 0 100 0.01)]
;                        ; draw a point with x = t * sin(t) and y = t * cos(t)
;                        (q/point (* t (q/sin t))
;                                 (* t (q/cos t))))))