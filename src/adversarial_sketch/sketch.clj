(ns adversarial-sketch.sketch)

"********* Description of the flags used by the sketch program"
"pragma options: passes flags to the compiler"
"--bnd-unroll-amnt: The synthesizer reasons about loops by unrolling them. This flag controls of the degree of unrolling."
"--bnd-arr-size: If an input array is dynamically sized, this flag can be used to control the maximum size."
"--bnd-int-range: Tells the synthesizer what range of values to consider for integer holes.  "
"--bnd-mbits: Tells the synthesizer how many bits to use to represent all bounds introduced by minimize(e) (default 5)"
(def source (format "pragma options \"--bnd-unroll-amnt 4 --bnd-arr1d-size 2 --bnd-arr-size 2 --bnd-int-range %d  --bnd-mbits %s\";

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

#include \"common.skh\"
bit renderSpecification(SHAPEVARIABLES) {
                                         assume shapeIdentity == CIRCLE || shapeIdentity == LINE || shapeIdentity == RECTANGLE;
                                         if (!HASCIRCLES) assume shapeIdentity != CIRCLE;
                                         if (!HASRECTANGLES) assume shapeIdentity != RECTANGLE;
                                         if (!HASLINES) assume shapeIdentity != LINE;
                                         else {
                                               if (!HASSOLID) assume dashed;
                                               if (!HASDASHED) assume !dashed;
                                               if (!HASARROW) assume !arrow;
                                               if (!HASNOARROW) assume arrow;
                                               }
                                         return %s;
                                         }
"                   biggestNumber
                    upperBoundOnLoss
                    (if usePrior "#define USEPRIOR" "")
                    maximumDepth
                    int(canLoop)
                    int(canReflect)
                    alreadyProvidedBounds
                    haveThisReflectionAlready
                    coefficientGenerator1
                    coefficientGenerator2
                    len(xCoefficients)
                    len(yCoefficients)
                    xValidation
                    yValidation
                    coefficientValidator1
                    coefficientValidator2
                    x1
                    y1
                    hasCircles
                    hasRectangles
                    hasLines
                    (True in solid)
                    (False in solid)
                    (True in arrows)
                    (False in arrows)
                    int(noDiagonals)
                    len(parse.lines)
                    " || ".join(parts)
                    ))

