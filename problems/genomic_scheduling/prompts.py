"""
EoH-compatible prompt specification for genomic pipeline scheduling.
"""

class GetPrompts():
    def __init__(self, scene_context=""):
        self.scene_context = scene_context

        self.prompt_task = (
            "I need help designing a novel priority function for online scheduling "
            "of genomic pipeline tasks. In each step, tasks are sorted by priority "
            "(higher = scheduled first) and assigned to available CPU resources. "
            "The goal is to minimize the makespan (maximum completion time). "
            "Each task has attributes: estimated_runtime, resource_need, weight, "
            "deadline, and task_type (string: 'alignment', 'variant_calling', or 'sv_calling')."
        )
        if scene_context:
            self.prompt_task += f"\n\nIMPORTANT CONTEXT: {scene_context}"

        self.prompt_func_name = "priority"
        self.prompt_func_inputs = ["task"]
        self.prompt_func_outputs = ["score"]
        self.prompt_inout_inf = (
            "'task' is an object with attributes: estimated_runtime (float), "
            "resource_need (float), weight (float), deadline (float), "
            "task_type (str). The output 'score' is a single float: "
            "higher score = higher scheduling priority."
        )
        self.prompt_other_inf = (
            "The function should be a pure Python function using only basic math "
            "operations (no numpy required). It should handle edge cases like "
            "zero or negative values gracefully. The function should be complex "
            "enough to balance multiple task attributes for good scheduling performance."
        )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
