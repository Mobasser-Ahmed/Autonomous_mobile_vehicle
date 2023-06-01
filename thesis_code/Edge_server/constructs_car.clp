(deftemplate state
  (slot sign)
  (slot distance)
  (slot speed))
(defrule rule1
    (state (sign "stop") (distance 1) (speed 1))
    =>
    (assert (result_speed: 0)))
(defrule rule2
    (state (sign "crosswalk") (distance 1) (speed 1))
    =>
    (assert (result_speed: 0.5)))
(defrule rule3
    (state (sign "") (distance 1) (speed 1))
    =>
    (assert (result_speed: 1)))