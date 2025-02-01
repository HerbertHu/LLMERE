TASK_DESC_SUBEVENT = "The current task is a subevent relation extraction task, which aims to identify " \
                     "subevent relations among events in texts. The subevent relation denotes a hierarchical " \
                     "relation where the first event is contained by the second, delineated into two subtypes: " \
                     "SuperSub and SubSuper. In the provided document, event trigger words are annotated within " \
                     "angle brackets (<>). The desired outcome is a list of events in the document that have " \
                     "a subevent relation with the given event. The prescribed output format should follow " \
                     "this structure: 'relation1: event1, event2; relation2: event3, event4'. " \
                     "The output 'relation: none' indicates that the given event lacks this " \
                     "particular type of relation with other events."
