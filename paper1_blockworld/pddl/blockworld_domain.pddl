(define (domain paper1-blockworld)
  (:requirements :typing :negative-preconditions :action-costs)

  (:types
    robot
    location
    block
    region)

  (:predicates
    (adjacent ?l1 - location ?l2 - location)
    (belongs ?l - location ?r - region)
    (at ?bot - robot ?l - location)
    (handempty ?bot - robot)
    (holding ?bot - robot ?b - block)
    (on ?b - block ?l - location)
    (clear ?l - location))

  (:functions
    (total-cost))

  (:action move
    :parameters (?bot - robot ?from - location ?to - location)
    :precondition (and
      (at ?bot ?from)
      (adjacent ?from ?to)
      (clear ?to))
    :effect (and
      (at ?bot ?to)
      (not (at ?bot ?from))
      (increase (total-cost) 25)))

  (:action pick
    :parameters (?bot - robot ?robotloc - location ?b - block ?blockloc - location)
    :precondition (and
      (at ?bot ?robotloc)
      (adjacent ?robotloc ?blockloc)
      (on ?b ?blockloc)
      (handempty ?bot))
    :effect (and
      (holding ?bot ?b)
      (not (handempty ?bot))
      (clear ?blockloc)
      (not (on ?b ?blockloc))
      (increase (total-cost) 100)))

  (:action place
    :parameters (?bot - robot ?robotloc - location ?b - block ?loc - location ?region - region)
    :precondition (and
      (holding ?bot ?b)
      (at ?bot ?robotloc)
      (adjacent ?robotloc ?loc)
      (clear ?loc)
      (belongs ?loc ?region))
    :effect (and
      (on ?b ?loc)
      (handempty ?bot)
      (not (holding ?bot ?b))
      (not (clear ?loc))
      (increase (total-cost) 100))))
