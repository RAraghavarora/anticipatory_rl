;; Clairvoyant K-task restaurant sequence domain.
;;
;; Companion domain to `toy_restaurant_domain.pddl`. Where the single-task
;; domain emits one PDDL problem per task and lets Fast Downward reason only
;; about that task, this domain reasons about a fixed chain of K tasks in one
;; search. The robot's physical world is shared across the chain; a state
;; machine on `is-current-task` enforces that each task must complete (in order)
;; before the next one starts.
;;
;; Design choices:
;;   - Types: `task_id` and `kind` are added so the state machine and the
;;     per-object kind/classification facts stay first-class.
;;   - Physical action semantics, names, signatures, and costs are preserved
;;     from `toy_restaurant_domain.pddl`. The only change is the universal
;;     `(not (current-task-satisfied))` precondition that gates them.
;;   - Six zero-cost `complete-*` actions advance the state machine once a task
;;     is satisfied. They are the only actions applicable after
;;     `current-task-satisfied` becomes true.
;;   - Deterministic consumption: witness predicates (`selected-*`) use object
;;     precedence to force Fast Downward to pick the first eligible cup/apple
;;     in object order, matching the env's `consume_delivery()`.
;;   - Water semantics: the abstract `water` constant is permanently tied to
;;     the fountain (it represents the always-present source at the fountain).
;;     Machine water availability is a separate first-class resource
;;     `(machine-water-available ?loc)` that `make-coffee` consumes and `pour`
;;     restores. This lets `fill` reach the concrete `water_fountain` object
;;     at the fountain while a normal state with `water_machine@coffeemachine`
;;     also still supports brewing at the machine.

(define (domain restaurant-sequence)

  (:requirements
    :strips
    :typing
    :adl
    :action-costs
    :derived-predicates
  )

  (:types
    location
    object
    task_id
    kind
  )

  (:constants
    water coffee - object
  )

  (:predicates
    ;; --- robot state ---
    (rob-at ?loc - location)
    (hand-is-free)
    (is-holding ?obj - object)

    ;; --- object state ---
    (is-at ?obj - object ?loc - location)
    (is-dirty ?obj - object)
    (is-pickable ?obj - object)
    (is-fillable ?obj - object)
    (is-jar ?obj - object)
    (is-liquid ?obj - object)
    (is-slicable ?obj - object)
    (is-container ?obj - object)
    (is-knife ?obj - object)
    (filled-with ?liquid - object ?container - object)
    (is-in ?inner - object ?outer - object)

    ;; --- location roles ---
    (is-fountain ?loc - location)
    (is-coffeemachine ?loc - location)
    (is-dishwasher ?loc - location)
    (is-countertop ?loc - location)

    ;; --- sequence state machine ---
    (is-current-task ?task - task_id)
    (next-task ?cur - task_id ?nxt - task_id)
    (current-task-satisfied)

    ;; --- task-type tags ---
    (task-is-serve-water ?task - task_id)
    (task-is-make-coffee ?task - task_id)
    (task-is-make-fruit-bowl ?task - task_id)
    (task-is-clear-containers ?task - task_id)
    (task-is-wash-objects ?task - task_id)
    (task-is-pick-place ?task - task_id)

    ;; --- task parameters ---
    (task-target-location ?task - task_id ?loc - location)
    (task-target-kind ?task - task_id ?kind - kind)
    (task-object ?task - task_id ?obj - object)

    ;; --- object classification (problem-defined) ---
    (object-kind ?obj - object ?kind - kind)
    (is-drink-container ?obj - object)
    (is-wash-ready ?loc - location)
    (object-precedes ?earlier - object ?later - object)

    ;; --- machine water resource (consumed by make-coffee, restored by pour) ---
    (machine-water-available ?loc - location)

    ;; --- determinism witnesses (derived) ---
    (selected-cup-for-serve-water ?task - task_id ?cup - object)
    (selected-cup-for-make-coffee ?task - task_id ?cup - object)
    (selected-apple-for-fruit-bowl ?task - task_id ?apple - object ?bowl - object)
  )

  (:functions (total-cost) (known-cost ?start ?end - location))

  ;; =========================================================================
  ;; DERIVED PREDICATES
  ;; =========================================================================

  ;; Deterministic cup witnesses for serve_water / make_coffee.
  ;; "The selected cup is the first object-precedes-eligible cup
  ;; at the task's target that is filled with the right liquid."
  (:derived (selected-cup-for-serve-water ?task - task_id ?cup - object)
    (exists (?loc - location)
      (and
        (is-current-task ?task)
        (task-is-serve-water ?task)
        (task-target-location ?task ?loc)
        (is-drink-container ?cup)
        (is-at ?cup ?loc)
        (filled-with water ?cup)
        (forall (?earlier - object)
          (or
            (not (is-drink-container ?earlier))
            (not (is-at ?earlier ?loc))
            (not (filled-with water ?earlier))
            (not (object-precedes ?earlier ?cup))
          )
        )
      )
    )
  )

  (:derived (selected-cup-for-make-coffee ?task - task_id ?cup - object)
    (exists (?loc - location)
      (and
        (is-current-task ?task)
        (task-is-make-coffee ?task)
        (task-target-location ?task ?loc)
        (is-drink-container ?cup)
        (is-at ?cup ?loc)
        (filled-with coffee ?cup)
        (forall (?earlier - object)
          (or
            (not (is-drink-container ?earlier))
            (not (is-at ?earlier ?loc))
            (not (filled-with coffee ?earlier))
            (not (object-precedes ?earlier ?cup))
          )
        )
      )
    )
  )

  ;; Deterministic apple witness for make_fruit_bowl.
  ;; The selected apple is the first object-precedes-eligible apple
  ;; that is contained in any container at the task target. The bowl
  ;; bound here is the apple's *actual* containing bowl, matching
  ;; Python's consume_delivery() which scans all apples (in object
  ;; order) across all qualifying bowls at the target -- not just
  ;; the earliest bowl.
  (:derived (selected-apple-for-fruit-bowl ?task - task_id ?apple - object ?bowl - object)
    (and
      (is-current-task ?task)
      (task-is-make-fruit-bowl ?task)
      (is-slicable ?apple)
      (is-container ?bowl)
      (is-in ?apple ?bowl)
      (exists (?loc - location)
        (and
          (task-target-location ?task ?loc)
          (is-at ?bowl ?loc)
        )
      )
      (forall (?earlier - object)
        (or
          (not (is-slicable ?earlier))
          (not (object-precedes ?earlier ?apple))
          (forall (?b - object)
            (or
              (not (is-container ?b))
              (not (is-in ?earlier ?b))
              (forall (?loc - location)
                (or
                  (not (task-target-location ?task ?loc))
                  (not (is-at ?b ?loc))
                )
              )
            )
          )
        )
      )
    )
  )

  ;; The current task is satisfied iff any of its satisfaction conditions
  ;; holds under the current state machine position. This is the single
  ;; derived predicate that gates every physical action.
  (:derived (current-task-satisfied)
    (or
      ;; serve_water: a cup/mug at the target contains water
      (exists (?task - task_id ?cup - object)
        (selected-cup-for-serve-water ?task ?cup)
      )
      ;; make_coffee: a cup/mug at the target contains coffee
      (exists (?task - task_id ?cup - object)
        (selected-cup-for-make-coffee ?task ?cup)
      )
      ;; make_fruit_bowl: a bowl at the target contains a sliced apple
      (exists (?task - task_id ?apple - object ?bowl - object)
        (selected-apple-for-fruit-bowl ?task ?apple ?bowl)
      )
      ;; clear_containers: no (is-at) object at the target AND no machine
      ;; water available at the target. The sequence init intentionally
      ;; omits the concrete (is-at water_machine ?loc) fact (machine water
      ;; is the abstract (machine-water-available ?loc) resource), so the
      ;; (forall ...) check alone would silently declare clear while the
      ;; executable env still sees a concrete water object at that
      ;; location. The extra (not (machine-water-available ?loc)) keeps the
      ;; sequence-domain and executable-env clear-containers semantics in
      ;; agreement both in the normal state and after `pour` re-supplies
      ;; the machine resource.
      (exists (?task - task_id ?loc - location)
        (and
          (is-current-task ?task)
          (task-is-clear-containers ?task)
          (task-target-location ?task ?loc)
          (forall (?obj - object) (not (is-at ?obj ?loc)))
          (not (machine-water-available ?loc))
        )
      )
      ;; wash_objects: an object of the target kind is clean, empty,
      ;; uncontained, at a wash-ready location
      (exists (?task - task_id ?obj - object ?loc - location ?k - kind)
        (and
          (is-current-task ?task)
          (task-is-wash-objects ?task)
          (task-target-kind ?task ?k)
          (object-kind ?obj ?k)
          (is-at ?obj ?loc)
          (is-wash-ready ?loc)
          (not (is-dirty ?obj))
          (not (filled-with water ?obj))
          (not (filled-with coffee ?obj))
          (forall (?b - object) (not (is-in ?obj ?b)))
        )
      )
      ;; pick_place: the fixed task object is at the target and hand is free
      (exists (?task - task_id ?obj - object ?loc - location)
        (and
          (is-current-task ?task)
          (task-is-pick-place ?task)
          (task-object ?task ?obj)
          (task-target-location ?task ?loc)
          (is-at ?obj ?loc)
          (hand-is-free)
        )
      )
    )
  )

  ;; =========================================================================
  ;; PHYSICAL ACTIONS
  ;; =========================================================================
  ;; Each physical action is identical to the single-task domain, plus the
  ;; universal gating precondition (not (current-task-satisfied)). When the
  ;; current task is satisfied, no physical action is applicable; the only
  ;; way forward is the matching `complete-*` action.

  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and
      (rob-at ?from)
      (not (= ?from ?to))
      (not (current-task-satisfied))
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
      (not (current-task-satisfied))
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
      (not (current-task-satisfied))
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
      (is-dirty ?obj)
      (not (current-task-satisfied))
    )
    :effect (and
      (not (is-dirty ?obj))
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
      (not (current-task-satisfied))
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
      ;; Aligned with the executable env: drain only requires that the held
      ;; container is water-filled at the fountain. The previous
      ;; `(is-fillable ?cnt)` rule was stricter than the env and rejected
      ;; e.g. a held water-filled plate, which the env accepts. Removed so
      ;; PDDL and env agree on the drain semantics.
      (filled-with water ?cnt)
      (not (current-task-satisfied))
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
      (not (current-task-satisfied))
    )
    :effect (and
      (not (filled-with ?liquid ?cnt))
      ;; Only water re-supplies the machine. Pouring coffee (or any other
      ;; non-water liquid) at the machine just empties the container.
      (when (= ?liquid water)
        (machine-water-available ?loc)
      )
      (increase (total-cost) 200)
    )
  )

  (:action make-coffee
    :parameters (?c - object ?loc - location)
    :precondition (and
      (rob-at ?loc)
      (is-coffeemachine ?loc)
      (is-at ?c ?loc)
      ;; Aligned with the env and the validator: only drink containers
      ;; (cup or mug) can be used for make-coffee. The previous
      ;; `(is-fillable ?c)` + `(not (is-jar ?c))` rule was too permissive
      ;; and let bowls and jars pass, even though the env rejects them.
      (is-drink-container ?c)
      (not (is-dirty ?c))
      (not (filled-with water ?c))
      (not (filled-with coffee ?c))
      (machine-water-available ?loc)
      (not (current-task-satisfied))
    )
    :effect (and
      (filled-with coffee ?c)
      (is-dirty ?c)
      (not (machine-water-available ?loc))
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
      ;; Free-object preconditions: the apple and bowl must not be
      ;; contained in anything. Aligned with the env's
      ;; `_is_action_valid` ("contained_in is None" for both the apple
      ;; and the bowl). The init emits `(is-in ?x ?y)` for any contained
      ;; object, so these guards are non-vacuous whenever the world has
      ;; containment.
      (forall (?b1 - object) (not (is-in ?a ?b1)))
      (is-container ?b)
      (is-at ?b ?loc)
      (forall (?b2 - object) (not (is-in ?b ?b2)))
      (not (is-dirty ?b))
      (not (filled-with water ?b))
      (not (filled-with coffee ?b))
      (not (current-task-satisfied))
    )
    :effect (and
      (not (is-at ?a ?loc))
      (is-in ?a ?b)
      (is-dirty ?b)
      (is-dirty ?k)
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
      (not (current-task-satisfied))
    )
    :effect (and
      (filled-with water ?cnt)
      (increase (total-cost) 50)
    )
  )

  ;; =========================================================================
  ;; COMPLETION ACTIONS (zero cost, advance the state machine)
  ;; =========================================================================
  ;; Each completion action:
  ;;   - Requires the corresponding task-type tag on the current task.
  ;;   - Requires a deterministic consumption witness (so the planner cannot
  ;;     "skip" or "delay" consumption by selecting a later eligible artifact).
  ;;   - Removes `is-current-task` for the current task and adds it for the
  ;;     successor.
  ;;   - Applies the task-specific consumption side effect.
  ;;   - Adds zero to (total-cost).

  (:action complete-serve-water
    :parameters (?cur - task_id ?nxt - task_id ?cup - object)
    :precondition (and
      (is-current-task ?cur)
      (next-task ?cur ?nxt)
      (task-is-serve-water ?cur)
      (current-task-satisfied)
      (selected-cup-for-serve-water ?cur ?cup)
    )
    :effect (and
      (not (is-current-task ?cur))
      (is-current-task ?nxt)
      (not (filled-with water ?cup))
    )
  )

  (:action complete-make-coffee
    :parameters (?cur - task_id ?nxt - task_id ?cup - object)
    :precondition (and
      (is-current-task ?cur)
      (next-task ?cur ?nxt)
      (task-is-make-coffee ?cur)
      (current-task-satisfied)
      (selected-cup-for-make-coffee ?cur ?cup)
    )
    :effect (and
      (not (is-current-task ?cur))
      (is-current-task ?nxt)
      (not (filled-with coffee ?cup))
    )
  )

  (:action complete-make-fruit-bowl
    :parameters (?cur - task_id ?nxt - task_id ?apple - object ?bowl - object ?loc - location)
    :precondition (and
      (is-current-task ?cur)
      (next-task ?cur ?nxt)
      (task-is-make-fruit-bowl ?cur)
      (task-target-location ?cur ?loc)
      (current-task-satisfied)
      (selected-apple-for-fruit-bowl ?cur ?apple ?bowl)
    )
    :effect (and
      (not (is-current-task ?cur))
      (is-current-task ?nxt)
      (not (is-in ?apple ?bowl))
      (is-at ?apple ?loc)
    )
  )

  (:action complete-clear-containers
    :parameters (?cur - task_id ?nxt - task_id)
    :precondition (and
      (is-current-task ?cur)
      (next-task ?cur ?nxt)
      (task-is-clear-containers ?cur)
      (current-task-satisfied)
    )
    :effect (and
      (not (is-current-task ?cur))
      (is-current-task ?nxt)
    )
  )

  (:action complete-wash-objects
    :parameters (?cur - task_id ?nxt - task_id)
    :precondition (and
      (is-current-task ?cur)
      (next-task ?cur ?nxt)
      (task-is-wash-objects ?cur)
      (current-task-satisfied)
    )
    :effect (and
      (not (is-current-task ?cur))
      (is-current-task ?nxt)
    )
  )

  (:action complete-pick-place
    :parameters (?cur - task_id ?nxt - task_id)
    :precondition (and
      (is-current-task ?cur)
      (next-task ?cur ?nxt)
      (task-is-pick-place ?cur)
      (current-task-satisfied)
    )
    :effect (and
      (not (is-current-task ?cur))
      (is-current-task ?nxt)
    )
  )
)
