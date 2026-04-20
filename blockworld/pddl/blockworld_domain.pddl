(define (domain paper1-blockworld)
  (:requirements :typing :negative-preconditions :action-costs)

  (:types
    robot
    location
    block
    region)

  (:predicates
    (move-edge ?l1 - location ?l2 - location)
    (adjacent ?l1 - location ?l2 - location)
    (belongs ?l - location ?r - region)
    (at ?bot - robot ?l - location)
    (handempty ?bot - robot)
    (holding ?bot - robot ?b - block)
    (on ?b - block ?l - location)
    (in-region ?b - block ?r - region)
    (clear ?l - location)
    (region-empty ?r - region))

  (:functions
    (total-cost)
    (move-cost ?from - location ?to - location))

  (:action move
    :parameters (?bot - robot ?from - location ?to - location)
    :precondition (and
      (at ?bot ?from)
      (move-edge ?from ?to))
    :effect (and
      (at ?bot ?to)
      (not (at ?bot ?from))
      (increase (total-cost) (move-cost ?from ?to))))

  (:action pick
    :parameters (?bot - robot ?robotloc - location ?b - block ?blockloc - location ?region - region)
    :precondition (and
      (at ?bot ?robotloc)
      (adjacent ?robotloc ?blockloc)
      (on ?b ?blockloc)
      (in-region ?b ?region)
      (belongs ?blockloc ?region)
      (handempty ?bot))
    :effect (and
      (holding ?bot ?b)
      (not (handempty ?bot))
      (clear ?blockloc)
      (region-empty ?region)
      (not (on ?b ?blockloc))
      (not (in-region ?b ?region))
      (increase (total-cost) 100)))

  (:action place
    :parameters (?bot - robot ?robotloc - location ?b - block ?loc - location ?region - region)
    :precondition (and
      (holding ?bot ?b)
      (at ?bot ?robotloc)
      (adjacent ?robotloc ?loc)
      (clear ?loc)
      (belongs ?loc ?region)
      (region-empty ?region))
    :effect (and
      (on ?b ?loc)
      (in-region ?b ?region)
      (handempty ?bot)
      (not (holding ?bot ?b))
      (not (clear ?loc))
      (not (region-empty ?region))
      (increase (total-cost) 100))))
