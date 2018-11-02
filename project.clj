(defproject adversarial-sketch "0.1.0-SNAPSHOT"
  :description "Learns Domain specific programs using data "
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :global-vars {*warn-on-reflection* true}
  :plugins [[lein-codox "0.10.3"]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [local/sketch-1.7.5-noarch "1.0"]
                 [quil "2.5.0"]
                 [uncomplicate/neanderthal "0.17.0"]
                 [org.clojure/tools.cli "0.3.5"]
                 [clojure-opennlp "0.4.0"]
                 [instaparse "1.4.8"]
                 [rhizome "0.2.9"]
                 [net.mikera/imagez "0.12.0"]
                 [thinktopic/think.image "0.4.16"]
                 [org.clojure/data.json "0.2.6"]
                 [tempfile "0.2.0"]]
  :jvm-opts ["-Xmx2g"]
  ;;:main ^:skip-aot adversarial-sketch.rp-lstm
   ;;:main ^:skip-aot adversarial-sketch.core
  :main ^:skip-aot adversarial-sketch.rp-lstm
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}
             :dev {:global-vars {*warn-on-reflection* true
                                 *unchecked-math* :warn-on-boxed}
                   :dependencies [[criterium "0.4.4"]]}}
  :repositories [["localrepo1" {:url "file:sketchrepo"
                                :username :env/localrepo_username
                                :password :env/localrepo_password}]])
