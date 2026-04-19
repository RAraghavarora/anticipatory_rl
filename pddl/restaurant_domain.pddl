(define (domain restaurant_symbolic)
  (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions :conditional-effects)
  (:types
    location object
  )

  (:predicates
    (adjacent ?l1 - location ?l2 - location)
    (agent-at ?l - location)
    (handfree)
    (holding ?o - object)
    (at ?o - object ?l - location)

    (mug-kind ?o - object)
    (glass-kind ?o - object)
    (bowl-kind ?o - object)

    (clean ?o - object)
    (empty ?o - object)
    (water ?o - object)
    (coffee ?o - object)
    (fruit ?o - object)

    (service-loc ?l - location)
    (wash-ready-loc ?l - location)
    (sink-loc ?l - location)
    (water-loc ?l - location)
    (coffee-loc ?l - location)
    (fruit-loc ?l - location)
  )

  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (agent-at ?from) (adjacent ?from ?to))
    :effect (and
      (agent-at ?to)
      (not (agent-at ?from))
    )
  )

  (:action pick
    :parameters (?o - object ?l - location)
    :precondition (and
      (handfree)
      (agent-at ?l)
      (at ?o ?l)
    )
    :effect (and
      (holding ?o)
      (not (handfree))
      (not (at ?o ?l))
    )
  )

  (:action place
    :parameters (?o - object ?l - location)
    :precondition (and
      (holding ?o)
      (agent-at ?l)
    )
    :effect (and
      (at ?o ?l)
      (handfree)
      (not (holding ?o))

      (when (and (sink-loc ?l) (not (empty ?o)))
        (and
          (not (clean ?o))
          (empty ?o)
          (not (water ?o))
          (not (coffee ?o))
          (not (fruit ?o))
        )
      )
      (when (and (service-loc ?l) (not (empty ?o)))
        (not (clean ?o))
      )
    )
  )

  (:action wash
    :parameters (?o - object ?l - location)
    :precondition (and
      (holding ?o)
      (agent-at ?l)
      (sink-loc ?l)
      (not (clean ?o))
    )
    :effect (and
      (clean ?o)
      (empty ?o)
      (not (water ?o))
      (not (coffee ?o))
      (not (fruit ?o))
    )
  )

  (:action fill-water
    :parameters (?o - object ?l - location)
    :precondition (and
      (holding ?o)
      (agent-at ?l)
      (water-loc ?l)
      (clean ?o)
      (empty ?o)
      (or (mug-kind ?o) (glass-kind ?o))
    )
    :effect (and
      (water ?o)
      (not (empty ?o))
      (not (coffee ?o))
      (not (fruit ?o))
    )
  )

  (:action brew-coffee
    :parameters (?o - object ?l - location)
    :precondition (and
      (holding ?o)
      (agent-at ?l)
      (coffee-loc ?l)
      (clean ?o)
      (empty ?o)
      (mug-kind ?o)
    )
    :effect (and
      (coffee ?o)
      (not (empty ?o))
      (not (water ?o))
      (not (fruit ?o))
    )
  )

  (:action fill-fruit
    :parameters (?o - object ?l - location)
    :precondition (and
      (holding ?o)
      (agent-at ?l)
      (fruit-loc ?l)
      (clean ?o)
      (empty ?o)
      (bowl-kind ?o)
    )
    :effect (and
      (fruit ?o)
      (not (empty ?o))
      (not (water ?o))
      (not (coffee ?o))
    )
  )
)
