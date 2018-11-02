(ns adversarial-sketch.sketch-parser
  (:require [instaparse.core :as insta]
            [clojure.string :as string]))

"This name space is parsing the training data and vectorizes."
"by parsing the values in image_traces.json (prepared in the form of the DSL syntax)
and mapping them to integer values "
(def trace-parser
  (insta/parser
    (clojure.java.io/resource "parser_vectorizer.bnf")))


(defn vectorizer [s]
  "Vectorizing a list of strings. Note that we have reserved 6 symbols already: 6+ range_of_coordinates
  Later we use End Symbol 22 and 24 more extra as guidance signals to sketch"
  (let [mystr s]
    (case mystr
      "Line" 1
      "Rectangle" 2
      "Circle" 3
      "True" 4
      "False" 5
      (+ (Integer. s) 6))))



(defn tracer-vectorizer [tokenized-strings]
  (->> (trace-parser tokenized-strings)
       (insta/transform
         {:expr identity
          :shape identity
          :x-coordinates1 identity
          :y-coordinates1 identity
          :x-coordinates2 identity
          :y-coordinates2 identity
          :radius identity
          :arrow identity
          :solid identity})
       (rest)
       (map #(vectorizer %))))


;(defn parse-expression [e]
;  (let [linear-exp
;  ;; Test a string against multiple regexps, and do something different
;  ;; with the match each time.
;  (condp re-matches e
;    #"([\-0-9]+) \* " :>> (fn [[_ p]]
;                            {:factor (Integer. p)})
;
;    #"\(-\(([0-9]+)\)\) \* " :>> (fn [[_ p]]
;                            {:factor (Integer. p)})
;
;
;    #" \+ ([\-0-9]+)" :>> (fn [[_ p]]
;                            {:offset (Integer. p)})
;
;    #"\[(\d)\]" :>> (fn [[_ p]]
;                      {:variable (get ["i" "j"] (Integer. p))}))]
;  linear-exp))


(defn search-regex [st regx]
  (re-find (re-pattern regx) st))


(defn parse-expression
  ;; Test a string against multiple regexps, and do something different
  ;; with the match each time.
  [e]
  (try {:factor 0 :variable nil :offset (Integer. e)} (catch Exception ex (let [m-positive-factor (search-regex e "([\\-0-9]+) \\* ")
                                              m-negative-factor (search-regex e "\\(-\\(([0-9]+)\\)\\) \\* ")
                                              m-offset (search-regex e " \\+ ([\\-0-9]+)")
                                              m-variable (search-regex e "\\[(\\d)\\]")
                                              factor (cond
                                                       m-positive-factor (Integer. (get m-positive-factor 1))
                                                       m-negative-factor (* -1 (Integer. (get m-negative-factor 1)))
                                                       :else 1)
                                              offset (if m-offset (Integer. (get m-offset 1)) 0)
                                              variable (if m-variable
                                                         (get ["i" "j"] (Integer. (get m-variable 1)))
                                                         nil)]
                                          {:factor factor :variable variable :offset offset}))))


;
;(defn get-block [output name starting-index]
;  (loop [output (drop starting-index output) d 0 starting-index starting-index]
;    (if (and (> d -1) (not (nil? (next output)) ))
;      (recur (next output)
;             (cond
;               (and (string/includes? (first output) name)
;                    (string/includes? (first output) "dummyStart")) (+ d 1)
;               (and (string/includes? (first output) name)
;                    (string/includes? (first output) "dummyEnd")) (- d 2)
;               :else d)
;             (+ 1 starting-index))
;      starting-index)))


(defn get-block
  [output name starting-index]
  (loop [d 0 starting-index starting-index]
    (if (and (> d -1))
      (recur (cond
               (and (string/includes? (get output starting-index) name)
                    (string/includes? (get output starting-index) "dummyStart")) (inc d)
               (and (string/includes? (get output starting-index) name)
                    (string/includes? (get output starting-index) "dummyEnd")) (dec d)
               :else d)
             (inc starting-index))
      starting-index)))




;(defn get-boundary [output starting-index]
;  (loop [cur-output output starting-index starting-index]
;    (cond
;      (string/includes? (first cur-output) "dummyStartBoundary") (get-block output "Boundary" (+ 1 starting-index))
;      (string/includes? (first cur-output) "dummyStartLoop") nil
;      (string/includes? (first cur-output) "dummyEndLoop") nil
;      :else (recur (next cur-output) (+ 1 starting-index)))))


(defn get-boundary [output starting-index]
  (loop [starting-index starting-index]
    (cond
      (string/includes? (get output starting-index) "dummyStartBoundary") (get-block output "Boundary" (inc starting-index))
      (string/includes? (get output starting-index) "dummyStartLoop") nil
      (string/includes? (get output starting-index) "dummyEndLoop") nil
      :else (recur (inc starting-index)))))


(defn apply-environment [st environment]
  (let [env (into (sorted-map-by (fn [key1 key2]
                                   (compare [(count key2) key2]
                                            [(count  key1) key1])))
                  environment)]
    (reduce #(apply string/replace %1 %2) st
            (seq (into [] env)))))



(defn coefficient-substitute [[st k] coefficient-index coefficient-value]
  (let [pattern (format "\\(coefficients%s[^\\[]*\\[%d\\]\\)" k coefficient-index)]
    (string/replace st (re-pattern pattern) (str coefficient-value))))



;(defn apply-coefficients
;  [st coefficients]
;  (if (string/includes? st "coefficients")
;    (loop [lp st
;           ks (keys coefficients)]
;      (if ks (recur (reduce
;                             #(apply coefficient-substitute %1 %2)
;                             [lp (first ks)] (into [] (map-indexed vector (get coefficients (first ks))))) (next ks)) lp)) st))
;
;

(defn apply-coefficients
  [st coefficients]
  (if (string/includes? st "coefficients")
    (loop [lp st
           ks (keys coefficients)]
      (if ks (recur (loop [lp lp
                                  coefficient-index 0
                                  coefficient-values (get coefficients (first ks))]
                             (if (next coefficient-values) (recur (coefficient-substitute [lp (first ks)] coefficient-index (first coefficient-values))
                                                                  (inc coefficient-index)
                                                                  (next coefficient-values) ) lp))
                           (next ks)) lp)) st))



(declare parse-sketch-output)

(defn get-reflection-commands [m-loop j & parse-sketch-output-args]
  (let [[output environment coefficients loop-depth commands] parse-sketch-output-args
        boundary-indx (get-boundary output (inc j))
        [j boundary environment coefficients] (if boundary-indx
                       (let [boundary (subvec output (inc j) boundary-indx)
                             results (parse-sketch-output boundary environment
                                                           coefficients (inc loop-depth))
                             coefficients (:coefficients results)
                             enviornment (:environment results)
                             boundary (:commands results)
                            ]
                         [boundary-indx boundary environment coefficients])
                       [j commands environment coefficients])
        body-index (get-block output "Loop" (inc j))
        body (subvec output (inc j) body-index)
        j body-index
        bound (parse-expression (get m-loop 2))
        result (parse-sketch-output body environment coefficients (inc loop-depth))
        body (:commands result)
        environment (:environment result)
        coefficients (:coefficients result)
        variable (get ["i" "j"] loop-depth)
        command [variable bound body boundary]
        ;commands (conj body command)
        ]
    [environment coefficients command j]))


(defn parse-sketch-output [output environment coefficients loop-depth]
  (loop [commands []
         environment (if environment environment {})
         coefficients (if coefficients coefficients {})
         output output
         j 0]
    (if (or (> j (- (count output) 1)) (string/includes? (get output j) "void renderSpecification"))
      {:commands commands :coefficients coefficients :environment environment}
      (let [m (search-regex (get output j) "validate[X|Y]\\((.*), (.*)\\)")]
        (if m
          (recur commands (assoc environment (get m 2) (get m 1)) coefficients output (inc j))
              (let [coefficients (let [m (search-regex (get output j) "int\\[[0-9]\\] coefficients([1|2]) = \\{([,0-9\\-]+)\\}" )]
                        (if m (assoc coefficients (Integer. (get m 1)) (mapv #(Integer. %) (string/split (get m 2) #",")  )) coefficients))
                    l (apply-environment (get output j) environment)
                    lp (apply-coefficients l coefficients)
                    m-circle (search-regex lp "\\(\\(shapeIdentity == 0\\) && \\(cx.* == (.+)\\)\\) && \\(cy.* == (.+)\\)")
                    m-line (search-regex lp "shapeIdentity == 1\\) && \\((.*) == lx1.*\\)\\) && \\((.*) == ly1.*\\)\\) && \\((.*) == lx2.*\\)\\) && \\((.*) == ly2.*\\)\\) && \\(([01]) == dashed\\)\\) && \\(([01]) == arrow")
                    m-rectangle (search-regex lp "shapeIdentity == 2\\) && \\((.+) == rx1.*\\)\\) && \\((.+) == ry1.*\\)\\) && \\((.+) == rx2.*\\)\\) && \\((.+) == ry2.*\\)")
                    m-reflection (search-regex lp "for\\(int (.*) = 0; .* < (.*); .* = .* \\+ 1\\)")
                    m-dummy-reflection (search-regex lp "dummyStartReflection\\(([0-9]+), ([0-9]+)\\)")]
                (cond
                  m-circle
                  (recur (conj commands ["circle" (parse-expression (get m-circle 1)) (parse-expression (get m-circle 2))])
                                  environment coefficients output (inc j))
                  m-line (recur (conj commands ["line" (parse-expression (get m-line 1)) (parse-expression (get m-line 2))
                                                  (parse-expression (get m-line 3)) (parse-expression (get m-line 4))
                                                  (format "arrow = %s" (= (get m-line 6) "1"))
                                                  (format "solid = %s" (= (get m-line 5) "0"))])
                                environment coefficients output (inc j))
                  m-rectangle (recur (conj commands ["rectangle" (parse-expression (get m-rectangle 1)) (parse-expression  (get m-rectangle 2) )
                                                         (parse-expression (get m-rectangle 3)) (parse-expression (get m-rectangle 4))])
                                       environment coefficients output (inc j))
                  (and m-reflection (not (string/includes? (get m-reflection 1) "reflectionIndex")))
                      (let [[environment coefficients commands j] (get-reflection-commands m-reflection j output environment coefficients loop-depth commands)]
                        (recur commands environment coefficients output j))
                  m-dummy-reflection (let [body-indx (get-block output "Reflection" (inc j))
                                           body (string/join "\n" (subvec output (inc j) body-indx))
                                           j body-indx
                                           x (Integer. (get m-dummy-reflection 1))
                                           y (Integer. (get m-dummy-reflection 2))
                                           axis (if (= y 0) "x" "y")
                                           coordinate (max x y)
                                           commands [axis coordinate (:commands (parse-sketch-output body environment coefficients loop-depth))]
                                           ] (recur commands environment coefficients output (inc j) ))
                  :else (recur commands environment coefficients output (inc j)))))))))





;(defn dsl-code-compiler
;  "The outpur of the above parser function parse-sketch-output
;    gets compiled to runnable code here"
;  [parsed-vectors parsed-dic]
;  (loop [parsed-vectors parsed-vectors
;         parsed-dict parsed-dict]
;    (if parsed-vectors (recur (next parsed-vectors)
;                               (let [node (get (first parsed-vectors) 0)]
;                                 (cond
;                                   (= node "i") (assoc parsed-dict )
;
;                                   ))))))
;

;
;
;
;(def compiler-parser-bnf
;  (insta/parser
;    (clojure.java.io/resource "graphic_compiler_parser.bnf")))
;
;(defn graphic-compiler-parser
;  "trims, compiles to the graphic-dsl via insta lib"
;  [sketch-ast]
;  (->> (clojure.string/replace sketch-ast #"[\[|\]|\"|\{|\}|,]|:|="  "")
;       compiler-parser-bnf))
;
;(defn visualize-ast
;  "visulize the ast"
;  [sketch-ast]
;  (insta/visualize (graphic-compiler-parser sketch-ast)) )
;
;
;(defn graphic-compiler-transformer [sketch-ast]
;  (->> (graphic-compiler-parser sketch-ast)
;       (insta/transform
;         {:loops
;          (fn [loop-var start-indx x loop-offset ] (format "(doseq [%s (range %s %s 1)]" loop-var start-indx loop-offset))
;          :circle
;          (fn [x-factor x-variable x-offset y-factor y-variable y-offset ] (format "(ellipse-draw (+ (* %s %s) %s) (+ (* %s %s) %s) 1)"
;                                                                                   x-factor (if (= x-variable "nil") 0 x-variable) x-offset
;                                                                                   y-factor (if (= y-variable "nil") 0 y-variable) y-offset
;                                                                                   ))
;          :line
;          (fn [x1-factor x1-variable x1-offset y1-factor y1-variable y1-offset
;               x2-factor x2-variable x2-offset y2-factor y2-variable y2-offset
;               arrow solid]
;            (format "(line-draw (+ (* %s %s) %s) (+ (* %s %s) %s) (+ (* %s %s) %s) (+ (* %s %s) %s))"
;                    x1-factor x1-variable x1-offset
;                    y1-factor y1-variable y1-offset
;                    x2-factor x2-variable x2-offset
;                    y2-factor y2-variable y2-offset))
;          :rectangle
;          (fn [x1-factor x1-variable x1-offset y1-factor y1-variable y1-offset
;               x2-factor x2-variable x2-offset y2-factor y2-variable y2-offset
;               arrow solid]
;            (format "(rectangular-draw (+ (* %s %s) %s) (+ (* %s %s) %s) (+ (* %s %s) %s) (+ (* %s %s) %s))"
;                    x1-factor x1-variable x1-offset
;                    y1-factor y1-variable y1-offset
;                    x2-factor x2-variable x2-offset
;                    y2-factor y2-variable y2-offset))})))
;
;
;
;(defn eval-graphic-draw [sketch-ast]
;  (->> (-> (graphic-compiler-transformer sketch-ast)
;           string/join
;           (str ")))"))
;       (str "(defn draw-magic []")
;       read-string
;       eval))
;

;
;(defn eval-graphic [sketch-ast]
;  (->> (-> (graphic-compiler-transformer sketch-ast)
;           string/join
;           (str "))"))
;       (str "(q/background 255)
;            (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)])")
;       read-string
;       eval))


;(defn eval-graphic-test [sketch-ast]
;  (->> (-> (graphic-compiler-transformer sketch-ast)
;           string/join
;           (str "))"))
;       (str
;         "(q/background 255)(q/with-translation [(/ (q/width) 2) (/ (q/height) 2)])")))
;
;
;(defn eval-graphic-test3 []
;  (-> (graphic-compiler-transformer sketch-ast)
;           string/join
;           (str "))")
;      read-string))


;(defn eval-graphic-defn [sketch-ast]
;  (->> (-> (graphic-compiler-transformer sketch-ast)
;           string/join
;           (str ")))"))
;       (str "(defn draw-new [] (q/background 255)(q/with-translation [(/ (q/width) 2) (/ (q/height) 2)])")))

;
;(defmacro draw-macro [sketch-ast]
;  `(~eval-graphic-test3 sketch-ast))
;
;
;(defn draw-new []
;  (draw-macro sketch-ast))


;(let [output-example "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpzqJj8W.sk:210\"; //Assume at tmpzqJj8W.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpzqJj8W.sk:212\"; //Assume at tmpzqJj8W.sk:212\n  assume (!(dashed)): \"Assume at tmpzqJj8W.sk:216\"; //Assume at tmpzqJj8W.sk:216\n  assume (!(arrow)): \"Assume at tmpzqJj8W.sk:217\"; //Assume at tmpzqJj8W.sk:217\n  int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)\n        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/\n        {\n          dummyStartBoundary();\n          bit _pac_sc_s26 = _pac_sc_s15_s17_0;\n          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n            _pac_sc_s26 = _pac_sc_s26_s30;\n          }\n          _pac_sc_s15_s17_0 = _pac_sc_s26;\n          dummyEndBoundary();\n        }\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39_1 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);\n          int y_s43_1 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1))"
;      ]
;  (parse-sketch-output output-example {} {} 0))

;
;(defn -main
;  ;  "I don't do a whole lot ... yet."
;  ;  (g/-main))
;  [& args]
;  (let [output-example
;        "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpzqJj8W.sk:210\"; //Assume at tmpzqJj8W.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpzqJj8W.sk:212\"; //Assume at tmpzqJj8W.sk:212\n  assume (!(dashed)): \"Assume at tmpzqJj8W.sk:216\"; //Assume at tmpzqJj8W.sk:216\n  assume (!(arrow)): \"Assume at tmpzqJj8W.sk:217\"; //Assume at tmpzqJj8W.sk:217\n  int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)\n        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/\n        {\n          dummyStartBoundary();\n          bit _pac_sc_s26 = _pac_sc_s15_s17_0;\n          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n            _pac_sc_s26 = _pac_sc_s26_s30;\n          }\n          _pac_sc_s15_s17_0 = _pac_sc_s26;\n          dummyEndBoundary();\n        }\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39_1 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);\n          int y_s43_1 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1))"
;        ;"void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpuV7thE.sk:217*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpuV7thE.sk:218\"; //Assume at tmpuV7thE.sk:218\n  assume (shapeIdentity != 0): \"Assume at tmpuV7thE.sk:219\"; //Assume at tmpuV7thE.sk:219\n  assume (shapeIdentity != 2): \"Assume at tmpuV7thE.sk:220\"; //Assume at tmpuV7thE.sk:220\n  assume (!(dashed)): \"Assume at tmpuV7thE.sk:224\"; //Assume at tmpuV7thE.sk:224\n  assume (!(arrow)): \"Assume at tmpuV7thE.sk:225\"; //Assume at tmpuV7thE.sk:225\n  int[0] environment = {};\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s17_s19 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpuV7thE.sk:104 (38)\n    bit _pac_sc_s33 = _pac_sc_s17_s19;\n    if(!(_pac_sc_s17_s19))/*tmpuV7thE.sk:111*/\n    {\n      int[1] _pac_sc_s33_s35 = {0};\n      push(0, environment, j, _pac_sc_s33_s35);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      bit _pac_sc_s17_s19_0 = 0;\n      for(int j_0 = 0; j_0 < 2; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpuV7thE.sk:104 (46)\n        bit _pac_sc_s33_0 = _pac_sc_s17_s19_0;\n        if(!(_pac_sc_s17_s19_0))/*tmpuV7thE.sk:111*/\n        {\n          int[2] _pac_sc_s33_s35_0 = {0,0};\n          push(1, _pac_sc_s33_s35, j_0, _pac_sc_s33_s35_0);\n          int x_s41 = 0;\n          validateX(((-(3)) * (_pac_sc_s33_s35_0[1])) + 5, x_s41);\n          int y_s45 = 0;\n          validateY((3 * (_pac_sc_s33_s35_0[0])) + 1, y_s45);\n          int x2_s49 = 0;\n          validateX(((-(3)) * (_pac_sc_s33_s35_0[1])) + 6, x2_s49);\n          int y2_s53 = 0;\n          validateY((3 * (_pac_sc_s33_s35_0[0])) + 1, y2_s53);\n          assert ((x_s41 == x2_s49) || (y_s45 == y2_s53)); //Assert at tmpuV7thE.sk:145 (234)\n          bit _pac_sc_s33_s37 = 0 || (((((((shapeIdentity == 1) && (x_s41 == lx1)) && (y_s45 == ly1)) && (x2_s49 == lx2)) && (y2_s53 == ly2)) && (0 == dashed)) && (0 == arrow));\n          int x_s41_0 = 0;\n          validateX(((-(3)) * (_pac_sc_s33_s35_0[0])) + 7, x_s41_0);\n          int y_s45_0 = 0;\n          validateY((3 * (_pac_sc_s33_s35_0[1])) + 2, y_s45_0);\n          int x2_s49_0 = 0;\n          validateX(((-(3)) * (_pac_sc_s33_s35_0[0])) + 7, x2_s49_0);\n          int y2_s53_0 = 0;\n          validateY((3 * (_pac_sc_s33_s35_0[1])) + 3, y2_s53_0);\n          assert ((x_s41_0 == x2_s49_0) || (y_s45_0 == y2_s53_0)); //Assert at tmpuV7thE.sk:145 (236)\n          loop_body_cost_0 = 2;\n          _pac_sc_s33_s37 = _pac_sc_s33_s37 || (((((((shapeIdentity == 1) && (x_s41_0 == lx1)) && (y_s45_0 == ly1)) && (x2_s49_0 == lx2)) && (y2_s53_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n          _pac_sc_s33_0 = _pac_sc_s33_s37;\n        }\n        _pac_sc_s17_s19_0 = _pac_sc_s33_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpuV7thE.sk:113 (30)\n      dummyEndLoop();\n      loop_body_cost = loop_body_cost_0 + 1;\n      _pac_sc_s33 = _pac_sc_s17_s19_0;\n    }\n    _pac_sc_s17_s19 = _pac_sc_s33;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpuV7thE.sk:113 (35)\n  dummyEndLoop();\n  _out = _pac_sc_s17_s19;\n  minimize(3 * (loop_body_cost + 1))"
;        ;"void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpuqIHtn.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpuqIHtn.sk:210\"; //Assume at tmpuqIHtn.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpuqIHtn.sk:212\"; //Assume at tmpuqIHtn.sk:212\n  assume (shapeIdentity != 1): \"Assume at tmpuqIHtn.sk:213\"; //Assume at tmpuqIHtn.sk:213\n  int[2] coefficients1 = {-3,24};\n  int[2] coefficients2 = {-3,16};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpuqIHtn.sk:96 (38)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpuqIHtn.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpuqIHtn.sk:96 (46)\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpuqIHtn.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, x_s39);\n          int y_s43 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, y_s43);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39)) && (cy == y_s43));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpuqIHtn.sk:105 (30)\n      dummyEndLoop();\n      loop_body_cost = loop_body_cost_0 + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpuqIHtn.sk:105 (35)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1))"
;        ;"int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    {\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        {\n          dummyStartBoundary();\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n          }\n          dummyEndBoundary();\n        }\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  minimize(3 * (loop_body_cost + 1))"
;        output-vectors (string/split-lines output-example)
;        ;"int ((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7_0 = 0"
;        ;"int ((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8_0 = 0"
;        ;coefficients {1 [-3 28] 2 [-3 24]}
;        ]
;    ;(print (apply-coefficients output-example coefficients)))
;    (print (:commands (parse-sketch-output output-vectors {} {} 0)))))
;    (graphic-dsl/)
;  ;(print (synthesize-program [2 6 7 8 9 2 7 8 9 10]))
;  ;(def output-example "void render (int shapeIdentity, int cx, int cy, int lx1, int ly1, int lx2, int ly2, bit dashed, bit arrow, int rx1, int ry1, int rx2, int ry2, ref bit _out)  implements renderSpecification/*tmpzqJj8W.sk:209*/\n{\n  _out = 0;\n  assume (((shapeIdentity == 0) || (shapeIdentity == 1)) || (shapeIdentity == 2)): \"Assume at tmpzqJj8W.sk:210\"; //Assume at tmpzqJj8W.sk:210\n  assume (shapeIdentity != 2): \"Assume at tmpzqJj8W.sk:212\"; //Assume at tmpzqJj8W.sk:212\n  assume (!(dashed)): \"Assume at tmpzqJj8W.sk:216\"; //Assume at tmpzqJj8W.sk:216\n  assume (!(arrow)): \"Assume at tmpzqJj8W.sk:217\"; //Assume at tmpzqJj8W.sk:217\n  int[2] coefficients1 = {-3,28};\n  int[2] coefficients2 = {-3,24};\n  int[0] environment = {};\n  int[1] coefficients1_0 = coefficients1[0::1];\n  int[1] coefficients2_0 = coefficients2[0::1];\n  dummyStartLoop();\n  int loop_body_cost = 0;\n  bit _pac_sc_s15_s17 = 0;\n  for(int j = 0; j < 3; j = j + 1)/*Canonical*/\n  {\n    assert (j < 4); //Assert at tmpzqJj8W.sk:96 (1334757887901394789)\n    bit _pac_sc_s31 = _pac_sc_s15_s17;\n    if(!(_pac_sc_s15_s17))/*tmpzqJj8W.sk:103*/\n    {\n      int[1] _pac_sc_s31_s33 = {0};\n      push(0, environment, j, _pac_sc_s31_s33);\n      dummyStartLoop();\n      int loop_body_cost_0 = 0;\n      int boundary_cost = 0;\n      bit _pac_sc_s15_s17_0 = 0;\n      for(int j_0 = 0; j_0 < 3; j_0 = j_0 + 1)/*Canonical*/\n      {\n        assert (j_0 < 4); //Assert at tmpzqJj8W.sk:96 (-4325113148049933570)\n        if(((j_0 > 0) && 1) && 1)/*tmpzqJj8W.sk:97*/\n        {\n          dummyStartBoundary();\n          bit _pac_sc_s26 = _pac_sc_s15_s17_0;\n          if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:99*/\n          {\n            int[2] _pac_sc_s26_s28 = {0,0};\n            push(1, _pac_sc_s31_s33, j_0, _pac_sc_s26_s28);\n            int x_s39 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 8, x_s39);\n            int y_s43 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y_s43);\n            int x2_s47 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[1])) + 9, x2_s47);\n            int y2_s51 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[0])) + 7, y2_s51);\n            assert ((x_s39 == x2_s47) || (y_s43 == y2_s51)); //Assert at tmpzqJj8W.sk:137 (2109344902378156491)\n            bit _pac_sc_s26_s30 = 0 || (((((((shapeIdentity == 1) && (x_s39 == lx1)) && (y_s43 == ly1)) && (x2_s47 == lx2)) && (y2_s51 == ly2)) && (0 == dashed)) && (0 == arrow));\n            int x_s39_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x_s39_0);\n            int y_s43_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 8, y_s43_0);\n            int x2_s47_0 = 0;\n            validateX(((coefficients1_0[0]) * (_pac_sc_s26_s28[0])) + 7, x2_s47_0);\n            int y2_s51_0 = 0;\n            validateY(((coefficients2_0[0]) * (_pac_sc_s26_s28[1])) + 9, y2_s51_0);\n            assert ((x_s39_0 == x2_s47_0) || (y_s43_0 == y2_s51_0)); //Assert at tmpzqJj8W.sk:137 (8471357942716875626)\n            boundary_cost = 2;\n            _pac_sc_s26_s30 = _pac_sc_s26_s30 || (((((((shapeIdentity == 1) && (x_s39_0 == lx1)) && (y_s43_0 == ly1)) && (x2_s47_0 == lx2)) && (y2_s51_0 == ly2)) && (0 == dashed)) && (0 == arrow));\n            _pac_sc_s26 = _pac_sc_s26_s30;\n          }\n          _pac_sc_s15_s17_0 = _pac_sc_s26;\n          dummyEndBoundary();\n        }\n        bit _pac_sc_s31_0 = _pac_sc_s15_s17_0;\n        if(!(_pac_sc_s15_s17_0))/*tmpzqJj8W.sk:103*/\n        {\n          int[2] _pac_sc_s31_s33_0 = {0,0};\n          push(1, _pac_sc_s31_s33, j_0, _pac_sc_s31_s33_0);\n          int x_s39_1 = 0;\n          validateX(((coefficients1_0[0]) * (_pac_sc_s31_s33_0[1])) + 7, x_s39_1);\n          int y_s43_1 = 0;\n          validateY(((coefficients2_0[0]) * (_pac_sc_s31_s33_0[0])) + 7, y_s43_1);\n          loop_body_cost_0 = 1;\n          _pac_sc_s31_0 = 0 || (((shapeIdentity == 0) && (cx == x_s39_1)) && (cy == y_s43_1));\n        }\n        _pac_sc_s15_s17_0 = _pac_sc_s31_0;\n      }\n      assert (loop_body_cost_0 != 0); //Assert at tmpzqJj8W.sk:105 (710966093749967188)\n      dummyEndLoop();\n      loop_body_cost = (loop_body_cost_0 + boundary_cost) + 1;\n      _pac_sc_s31 = _pac_sc_s15_s17_0;\n    }\n    _pac_sc_s15_s17 = _pac_sc_s31;\n  }\n  assert (loop_body_cost != 0); //Assert at tmpzqJj8W.sk:105 (-6090248756724217227)\n  dummyEndLoop();\n  _out = _pac_sc_s15_s17;\n  minimize(3 * (loop_body_cost + 1))")
;







;(defn tree-edit
;  "borrowd from https://www.ibm.com/developerworks/library/j-treevisit/"
;  [zipper matcher editor]
;  (loop [loc zipper]
;    (if (zip/end? loc)
;      (zip/root loc)
;      (if-let [matcher-result (matcher (zip/node loc))]
;        (recur (zip/next (zip/edit loc (partial editor matcher-result))))
;        (recur (zip/next loc))))))



;(defn chars-to-int [s]
;  "although strings in clojure are seqable but for numbers there is only one dumb way"
;  (map (fn [^Character c] (Character/digit c 10)) s))


;(def as-and-bs
;  (insta/parser
;    "S = AB*
;     AB = A B
;     A = 'a'+
;     B = 'b'+"))
;


;(def as-and-bs
;  (insta/parser
;    (clojure.java.io/resource "myparser.bnf")))
;
;(as-and-bs "aaaaabbbaaaabb")
;
;(insta/visualize (as-and-bs "aaabbab"))


;(def arithmetic
;  (insta/parser
;    (clojure.java.io/resource "anotherparser.bnf")))
;
;(->> (arithmetic "1-2/(3-4)+5*6")
;     (insta/transform
;       {:add +, :sub -, :mul *, :div /,
;        :number clojure.edn/read-string :expr identity}))
;
;(insta/visualize (arithmetic "1-2/(3-4)+5*6"))

;
;(defn trace-coordinate-transform [s]
;  "Helper function for transorming trace coordinates"
;  (cond
;    (= 1 (count s))  (Integer/parseInt s)
;    (= 2 (count s))  (let [x1 (subs s 0 1)
;                           x2 (subs s 1 2)]
;                       (+ (* (Integer/parseInt x1) 16) (Integer/parseInt x2)))
;    (= 3 (count s))  (let [x1 (subs s 0 2)
;                           x2 (subs s 2 3)]
;                       (if (< (Integer/parseInt x1) 16) (+ (* (Integer/parseInt x1) 16) (Integer/parseInt x2))
;                                                        (let [x1 (subs s 0 1)
;                                                              x2 (subs s 1 3)]
;                                                          (+ (* (Integer/parseInt x1) 16) (Integer/parseInt x2)))))
;    (= 4 (count s))  (let [x1 (subs s 0 2)
;                           x2 (subs s 2 4)]
;                       (+ (* (Integer/parseInt x1) 16) (Integer/parseInt x2)))
;    :else nil))
