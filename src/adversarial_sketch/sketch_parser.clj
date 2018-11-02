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
