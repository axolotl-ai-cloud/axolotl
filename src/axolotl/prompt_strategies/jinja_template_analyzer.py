"""Module for inspect jinja templates for the variables they use"""

from typing import Dict, Optional, Set, TypedDict, Union

from jinja2 import Environment, meta, nodes


class JinjaTemplateAnalysis(TypedDict):
    """
    Represents the detailed analysis of a Jinja template variable.

    Attributes:
        accessed_properties (Set[str]): A set of properties accessed from the variable
            (e.g., `foo.bar` results in 'bar' being accessed for 'foo').
        accessed_indices (Set[Union[int, float]]): A set of indices accessed from the variable.
        is_iterated (bool): Indicates if the variable is used as an iteration source in a `for` loop.
        is_conditional (bool): Indicates if the variable is referenced within a conditional statement (e.g., an `if` block).
        iteration_source (Optional[str]): The name of the variable being iterated over, if applicable.
        iteration_target (Optional[Union[str, list[str]]]): The loop target(s) assigned in the iteration.
    """

    accessed_properties: Set[str]
    accessed_indices: Set[Union[int, float]]
    is_iterated: bool
    is_conditional: bool
    iteration_source: Optional[str]
    iteration_target: Optional[Union[str, list[str]]]


class JinjaTemplateAnalyzer:
    """
    Analyzes Jinja templates to extract information about variable usage,
    including accessed properties, iteration, and conditional references.

    Attributes:
        env (jinja2.Environment): The Jinja2 environment used for parsing templates.
        property_access (Dict[str, Set[str]]): Tracks accessed properties for variables.
        iteration_targets (Dict[str, str]): Maps iteration target variables to their sources.

    Methods:
        get_template_variables(template: str) -> Dict[str, Set[str]]:
            Parse a Jinja template and return a mapping of variables to their accessed properties.

        analyze_template(template: str) -> Dict[str, JinjaTemplateAnalysis]:
            Perform a detailed analysis of the template, including variable usage,
            iteration, and conditional references.

    Private Methods:
        _visit_node(node) -> None:
            Recursively visit AST nodes to detect attribute access and iteration targets.

        _get_base_name(node) -> Optional[str]:
            Extract the base variable name from a node.

        _get_target_name(node) -> Optional[Union[str, list[str]]]:
            Extract the target name(s) from a `For` node.
    """

    def __init__(self, template: str):
        self.env: Environment = Environment(autoescape=True)
        self.property_access: Dict[str, Set[str]] = {}
        self.iteration_targets: Dict[str, Union[str, list[str]]] = {}
        self.index_access: Dict[str, Set[Union[int, float]]] = {}
        self.ast: nodes.Node = self.env.parse(template)
        self.template: str = template
        self.variable_assignments: Dict[str, str] = {}

    def _visit_node(self, node) -> None:
        """Recursively visit AST nodes to find attribute access."""
        # Handle attribute access (dot notation)
        if isinstance(node, nodes.Getattr):
            base_name = self._get_base_name(node.node)
            if base_name:
                self.property_access.setdefault(base_name, set()).add(node.attr)

        # Handle dictionary access (subscript notation)
        elif isinstance(node, nodes.Getitem):
            base_name = self._get_base_name(node.node)
            if base_name and isinstance(node.arg, nodes.Const):
                value = node.arg.value
                if isinstance(value, (int, float)):
                    self.index_access.setdefault(base_name, set()).add(value)
                else:
                    self.property_access.setdefault(base_name, set()).add(value)

        elif isinstance(node, nodes.Test) and node.name == "defined":
            base_name = self._get_base_name(node.node)
            if base_name:
                if isinstance(node.node, nodes.Getattr):
                    self.property_access.setdefault(base_name, set()).add(
                        node.node.attr
                    )

        # Handle loop variables
        elif isinstance(node, nodes.For):
            iter_name = self._get_base_name(node.iter)
            target_name = self._get_target_name(node.target)
            if iter_name and target_name:
                self.iteration_targets[target_name] = iter_name
                self.property_access.setdefault(iter_name, set())

        elif isinstance(node, nodes.Assign):
            target_name = self._get_target_name(node.target)
            source_name = self._get_base_name(node.node)
            if target_name and source_name:
                self.variable_assignments[target_name] = source_name

        elif isinstance(node, nodes.Filter):
            if node.name == "selectattr":
                target = self._get_base_name(node.node)
                if target:
                    self.variable_assignments[f"filtered_{target}"] = target

        for child in node.iter_child_nodes():
            self._visit_node(child)

    def _get_target_name(self, node) -> Optional[str]:
        """Get the target variable name from a For node.

        Args:
            node: A Jinja AST node representing either a Name or Tuple node

        Returns:
            - str: For simple variable targets (e.g., "item" in "for item in items")
            - None: If the node type is not recognized or is a tuple
        """
        if isinstance(node, nodes.Name):
            return node.name
        return None

    def _get_target_names(self, node) -> list[str]:
        """Get all target variable names from a For node, including tuple unpacking.

        Args:
            node: A Jinja AST node representing either a Name or Tuple node

        Returns:
            List of target variable names
        """
        if isinstance(node, nodes.Name):
            return [node.name]

        if isinstance(node, nodes.Tuple):
            names = []
            for n in node.items:
                if isinstance(n, nodes.Name):
                    names.append(n.name)
            return names

        return []

    def _get_base_name(self, node) -> Optional[str]:
        """Get the base variable name from a node."""
        if isinstance(node, nodes.Name):
            return node.name

        if isinstance(node, nodes.Getattr):
            return self._get_base_name(node.node)

        if isinstance(node, nodes.Getitem):
            return self._get_base_name(node.node)

        return None

    def get_template_variables(self) -> Dict[str, Set[str]]:
        """
        Parse a Jinja template and return both variables and their accessed properties.

        Args:
            template (str): The Jinja template string

        Returns:
            Dict[str, Set[str]]: Dictionary mapping variable names to sets of accessed properties
        """
        # Parse the template
        ast = self.env.parse(self.template)

        # Get all undeclared variables
        variables = meta.find_undeclared_variables(ast)

        # Reset property access tracking
        self.property_access = {}

        # Visit all nodes to find property access
        self._visit_node(ast)

        # Create result dictionary
        result: Dict[str, Set[str]] = {var: set() for var in variables}
        # Merge in any discovered sub-properties
        for var, props in self.property_access.items():
            if var not in result:
                result[var] = set()
            result[var].update(props)

        return result

    def analyze_template(self) -> Dict[str, JinjaTemplateAnalysis]:
        """
        Provide a detailed analysis of template variables and their usage.
        """
        variables = self.get_template_variables()
        self.iteration_targets = {}

        analysis: Dict[str, JinjaTemplateAnalysis] = {
            var: JinjaTemplateAnalysis(
                accessed_properties=props,
                accessed_indices=set(),
                is_iterated=False,
                is_conditional=False,
                iteration_source=None,
                iteration_target=None,
            )
            for var, props in variables.items()
        }

        for var, indices in self.index_access.items():
            if var in analysis:
                analysis[var]["accessed_indices"] = indices

        def visit_node(node):
            if isinstance(node, nodes.If):

                def find_test_vars(test_node):
                    if isinstance(test_node, nodes.Name):
                        if test_node.name in analysis:
                            analysis[test_node.name]["is_conditional"] = True
                    for child in test_node.iter_child_nodes():
                        find_test_vars(child)

                find_test_vars(node.test)

            if isinstance(node, nodes.For):
                iter_target = self._get_base_name(node.iter)
                target_name = self._get_target_name(node.target)
                if iter_target in analysis:
                    analysis[iter_target]["is_iterated"] = True
                    if target_name:
                        analysis[iter_target]["iteration_target"] = target_name
                        if isinstance(target_name, str) and target_name not in analysis:
                            analysis[target_name] = {
                                "accessed_properties": set(),
                                "is_iterated": False,
                                "is_conditional": False,
                                "iteration_source": iter_target,
                                "iteration_target": None,
                            }

            for child in node.iter_child_nodes():
                visit_node(child)

        visit_node(self.ast)
        return analysis

    def get_downstream_properties(self, start_var: str) -> Dict[str, Set[str]]:
        """
        Get all properties accessed on a variable and its downstream assignments.

        Args:
            start_var: The starting variable to trace

        Returns:
            Dict mapping variable names to their accessed properties
        """
        visited = set()
        properties = {}

        def trace_variable(var_name: str):
            if var_name in visited:
                return
            visited.add(var_name)

            # Get direct properties
            if var_name in self.property_access:
                properties[var_name] = self.property_access[var_name]

            # Get properties from iteration targets
            if var_name in self.iteration_targets:
                target = self.iteration_targets[var_name]
                if isinstance(target, str):
                    trace_variable(target)
                elif isinstance(target, list):
                    for t in target:
                        trace_variable(t)

            # Follow assignments
            for target, source in self.variable_assignments.items():
                if source == var_name:
                    trace_variable(target)

            # Check for array slicing
            analysis = self.analyze_template()
            if var_name in analysis:
                var_info = analysis[var_name]
                if var_info["accessed_indices"]:
                    # If this variable is sliced, follow the resulting assignment
                    slice_result = f"{var_name}_slice"
                    if slice_result in self.property_access:
                        trace_variable(slice_result)

        trace_variable(start_var)
        return properties

    def get_message_vars(self, field_messages: str = "messages") -> Set[str]:
        """
        Get all properties accessed on messages and derived variables.
        """
        all_properties = self.get_downstream_properties(field_messages)

        # Combine all properties from all related variables
        combined_properties = set()
        for properties in all_properties.values():
            combined_properties.update(properties)

        # Also include properties from the message iteration variable
        analysis = self.analyze_template()
        if "message" in analysis:
            combined_properties.update(analysis["message"]["accessed_properties"])

        return combined_properties
