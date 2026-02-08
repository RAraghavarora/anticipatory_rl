(define (domain gridworld-rearrangement)
  (:requirements :typing :negative-preconditions :conditional-effects)
  (:types
    agent
    support
      location cargo - support
    receptacle)

  (:predicates
    (adjacent ?l1 - location ?l2 - location)
    (agent-at ?a - agent ?l - location)
    (handfree ?a - agent)
    (holding ?a - agent ?o - cargo)
    (on ?o - cargo ?s - support)
    (clear ?s - support)
    (belongs ?l - location ?r - receptacle)
    (in ?o - cargo ?r - receptacle)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adjacent ?from ?to))
    :effect (and (agent-at ?a ?to)
                 (not (agent-at ?a ?from))))

  (:action pick-from-location
    :parameters (?a - agent ?obj - cargo ?loc - location ?rec - receptacle)
    :precondition (and (agent-at ?a ?loc)
                       (on ?obj ?loc)
                       (clear ?obj)
                       (handfree ?a)
                       (belongs ?loc ?rec)
                       (in ?obj ?rec))
    :effect (and (holding ?a ?obj)
                 (not (handfree ?a))
                 (clear ?loc)
                 (not (on ?obj ?loc))
                 (not (in ?obj ?rec))))

  (:action pick-from-stack
    :parameters (?a - agent ?obj - cargo ?base - cargo ?loc - location ?rec - receptacle)
    :precondition (and (agent-at ?a ?loc)
                       (on ?obj ?base)
                       (on ?base ?loc)
                       (clear ?obj)
                       (handfree ?a)
                       (in ?base ?rec)
                       (in ?obj ?rec))
    :effect (and (holding ?a ?obj)
                 (not (handfree ?a))
                 (clear ?base)
                 (not (on ?obj ?base))
                 (not (in ?obj ?rec))))

  (:action place-on-location
    :parameters (?a - agent ?obj - cargo ?loc - location ?rec - receptacle)
    :precondition (and (holding ?a ?obj)
                       (agent-at ?a ?loc)
                       (clear ?loc)
                       (belongs ?loc ?rec))
    :effect (and (handfree ?a)
                 (not (holding ?a ?obj))
                 (on ?obj ?loc)
                 (not (clear ?loc))
                 (clear ?obj)
                 (in ?obj ?rec)))

  (:action stack-on-object
    :parameters (?a - agent ?obj - cargo ?base - cargo ?loc - location ?rec - receptacle)
    :precondition (and (holding ?a ?obj)
                       (agent-at ?a ?loc)
                       (on ?base ?loc)
                       (clear ?base)
                       (in ?base ?rec))
    :effect (and (handfree ?a)
                 (not (holding ?a ?obj))
                 (on ?obj ?base)
                 (not (clear ?base))
                 (clear ?obj)
                 (in ?obj ?rec))))
