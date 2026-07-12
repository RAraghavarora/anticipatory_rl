(define (domain restaurant)
  (:requirements :strips :typing :action-costs)

  (:types location object)

  (:constants
    water coffee - object
  )

  (:predicates
    (rob-at ?loc - location)
    (hand-is-free)
    (is-holding ?obj - object)
    (is-at ?obj - object ?loc - location)
    (is-dirty ?obj - object)
    (is-pickable ?obj - object)
    (is-fillable ?obj - object)
    (is-jar ?obj - object)
    (is-liquid ?obj - object)
    (is-slicable ?obj - object)
    (is-container ?obj - object)
    (is-knife ?obj - object)
    (is-coffeegrinds ?obj - object)
    (is-washable ?obj - object)
    (is-fountain ?loc - location)
    (is-coffeemachine ?loc - location)
    (is-dishwasher ?loc - location)
    (is-countertop ?loc - location)
    (filled-with ?liquid - object ?container - object)
    (is-in ?apple - object ?bowl - object)
  )

  (:functions (total-cost) (known-cost ?start ?end - location))

  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and
      (rob-at ?from)
      (not (= ?from ?to))
    )
    :effect (and
      (not (rob-at ?from))
      (rob-at ?to)
      (increase (total-cost) (known-cost ?from ?to))
    )
  )

  (:action pick
    :parameters (?obj - object ?loc - location)
    :precondition (and
      (hand-is-free)
      (rob-at ?loc)
      (is-at ?obj ?loc)
      (is-pickable ?obj)
    )
    :effect (and
      (not (hand-is-free))
      (is-holding ?obj)
      (not (is-at ?obj ?loc))
      (increase (total-cost) 100)
    )
  )

  (:action place
    :parameters (?obj - object ?loc - location)
    :precondition (and
      (is-holding ?obj)
      (rob-at ?loc)
    )
    :effect (and
      (hand-is-free)
      (not (is-holding ?obj))
      (is-at ?obj ?loc)
      (increase (total-cost) 100)
    )
  )

  (:action wash
    :parameters (?obj - object ?loc - location)
    :precondition (and
      (rob-at ?loc)
      (is-dishwasher ?loc)
      (is-at ?obj ?loc)
      (is-washable ?obj)
      (is-dirty ?obj)
    )
    :effect (and
      (not (is-dirty ?obj))
      (not (filled-with water ?obj))
      (not (filled-with coffee ?obj))
      (increase (total-cost) 200)
    )
  )

  (:action fill
    :parameters (?cnt - object ?loc - location ?src - object)
    :precondition (and
      (rob-at ?loc)
      (is-fountain ?loc)
      (is-at ?src ?loc)
      (is-liquid ?src)
      (is-fillable ?cnt)
      (is-holding ?cnt)
      (not (is-dirty ?cnt))
      (not (filled-with water ?cnt))
      (not (filled-with coffee ?cnt))
    )
    :effect (and
      (filled-with water ?cnt)
      (increase (total-cost) 1000)
    )
  )

  (:action drain
    :parameters (?cnt - object ?loc - location)
    :precondition (and
      (rob-at ?loc)
      (is-fountain ?loc)
      (is-holding ?cnt)
      (is-fillable ?cnt)
      (filled-with water ?cnt)
    )
    :effect (and
      (not (filled-with water ?cnt))
      (increase (total-cost) 50)
    )
  )

  (:action pour
    :parameters (?cnt - object ?liquid - object ?loc - location)
    :precondition (and
      (rob-at ?loc)
      (is-coffeemachine ?loc)
      (is-liquid ?liquid)
      (filled-with ?liquid ?cnt)
      (is-holding ?cnt)
    )
    :effect (and
      (not (filled-with ?liquid ?cnt))
      (is-at ?liquid ?loc)
      (increase (total-cost) 200)
    )
  )

  (:action make-coffee
    :parameters (?c - object ?loc - location)
    :precondition (and
      (rob-at ?loc)
      (is-coffeemachine ?loc)
      (is-at ?c ?loc)
      (is-fillable ?c)
      (not (is-jar ?c))
      (not (is-dirty ?c))
      (not (filled-with water ?c))
      (not (filled-with coffee ?c))
      (is-at water ?loc)
    )
    :effect (and
      (filled-with coffee ?c)
      (is-dirty ?c)
      (not (is-at water ?loc))
      (increase (total-cost) 50)
    )
  )

  (:action make-fruit-bowl
    :parameters (?a - object ?b - object ?k - object ?loc - location)
    :precondition (and
      (rob-at ?loc)
      (is-countertop ?loc)
      (is-holding ?k)
      (is-knife ?k)
      (not (is-dirty ?k))
      (is-slicable ?a)
      (is-at ?a ?loc)
      (is-container ?b)
      (is-at ?b ?loc)
      (not (is-dirty ?b))
      (not (filled-with water ?b))
      (not (filled-with coffee ?b))
    )
    :effect (and
      (not (is-at ?a ?loc))
      (is-in ?a ?b)
      (is-dirty ?k)
      (is-dirty ?b)
      (increase (total-cost) 100)
    )
  )

  (:action refill_water
    :parameters (?cnt - object ?loc - location ?jr - object)
    :precondition (and
      (rob-at ?loc)
      (is-at ?jr ?loc)
      (is-holding ?cnt)
      (is-jar ?jr)
      (is-fillable ?cnt)
      (not (is-dirty ?cnt))
      (not (filled-with water ?cnt))
      (not (filled-with coffee ?cnt))
      (filled-with water ?jr)
    )
    :effect (and
      (filled-with water ?cnt)
      (increase (total-cost) 50)
    )
  )
)
