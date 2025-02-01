TASK_DESC_TEMPORAL = "The current task is an event temporal relation extraction task, which aims to identify temporal " \
                     "relations among events in texts. The temporal relation between events refers to the " \
                     "chronological order in which they occur, involving six subtypes, namely, SIMULTANEOUS, ENDS-ON, " \
                     "BEGINS-ON, OVERLAP, CONTAINS, and BEFORE. In the provided document, event trigger words are " \
                     "annotated within angle brackets (<>). The desired outcome is a list of events in the document " \
                     "that have temporal relations with the given event. The prescribed output format should follow " \
                     "this structure: 'relation1: event1, event2; relation2: event3, event4'. The output 'relation: " \
                     "none' indicates that the given event lacks this particular type of relation with other events."

TASK_DESC_CAUSAL = "The current task is an event causal relation extraction task, which aims to identify causal " \
                   "relations among events in texts. The causal relation between events denotes that the occurrence " \
                   "of the first event precipitates the happening of the second event, delineated into two subtypes: " \
                   "CAUSE and PRECONDITION. In the provided document, event trigger words are annotated within angle " \
                   "brackets (<>). The desired outcome is a list of events in the document that have causal relations " \
                   "with the given event. The prescribed output format should follow this structure: 'relation1: " \
                   "event1, event2; relation2: event3, event4'. The output 'relation: none' indicates that the given " \
                   "event lacks this particular type of relation with other events."

TASK_DESC_SUBEVENT = "The current task is a subevent relation extraction task, which aims to identify subevent " \
                     "relations among events in texts. The subevent relation, labeled as SUBEVENT, denotes a " \
                     "hierarchical relation where the first event is contained by the second. In the provided " \
                     "document, event trigger words are annotated within angle brackets (<>). The desired outcome is " \
                     "a list of events in the document that have a subevent relation with the given event. The " \
                     "prescribed output format should follow this structure: 'relation1: event1, event2; relation2: " \
                     "event3, event4'. The output 'relation: none' indicates that the given event lacks this " \
                     "particular type of relation with other events."

TASK_DESC_COREF = "The current task is an event coreference relation extraction task, which aims to identify " \
                  "coreference relations among events in texts. The coreference relation, labeled as COREFERENCE, " \
                  "denotes that two events are the same one. In the provided document, event trigger words are " \
                  "annotated within angle brackets (<>). The desired outcome is a list of events in the document that " \
                  "have a coreference relation with the given event. The prescribed output format should follow this " \
                  "structure: 'relation1: event1, event2; relation2: event3, event4'. The output 'relation: none' " \
                  "indicates that the given event lacks this particular type of relation with other events."
