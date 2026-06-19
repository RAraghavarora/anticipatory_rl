(define (problem toy-restaurant-1)
  (:domain restaurant)

  (:objects
    counter coffee_bar sink dishwasher table - location
    cup glass apple bowl knife grinds - object
  )

  (:init
    ; robot
    (rob-at counter)
    (hand-is-free)

    ; locations
    (is-countertop counter)
    (is-coffeemachine coffee_bar)
    (is-fountain sink)
    (is-dishwasher dishwasher)

    ; adjacency
    (adjacent counter table)
    (adjacent table counter)
    (adjacent counter sink)
    (adjacent sink counter)
    (adjacent counter coffee_bar)
    (adjacent coffee_bar counter)
    (adjacent sink dishwasher)
    (adjacent dishwasher sink)

    ; object locations
    (is-at cup counter)
    (is-at glass sink)
    (is-at apple counter)
    (is-at bowl counter)
    (is-at knife counter)
    (is-at grinds coffee_bar)
    (is-at water sink)

    ; object properties
    (is-pickable cup)
    (is-pickable glass)
    (is-pickable apple)
    (is-pickable bowl)
    (is-pickable knife)
    (is-pickable grinds)

    (is-fillable cup)
    (is-fillable glass)
    (is-container bowl)
    (is-knife knife)
    (is-slicable apple)
    (is-coffeegrinds grinds)
    (is-liquid water)

    (is-washable cup)
    (is-washable glass)
    (is-washable bowl)
    (is-washable knife)
  )

  (:goal
    (and
      (filled-with water glass)
      (is-at glass table)
    )
  )

  (:metric minimize (total-cost))
)
