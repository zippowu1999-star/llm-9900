
class ReWoorRuntime:
    def __init__(self, tool_registry: dict, logger=print):
        self.tool_registry = tool_registry
        self.log = logger

    def _resolve_arg(self, arg, context):
        # Resolve references like "#E3" or "#E8.train"
        if isinstance(arg, str) and arg.startswith("#E"):
            if "." in arg:
                root, sub = arg.split(".", 1)
                obj = context.get(root)
                if isinstance(obj, dict):
                    return obj.get(sub)
                if hasattr(obj, "get"):
                    return obj.get(sub)
                return getattr(obj, sub)
            return context.get(arg)
        if isinstance(arg, dict):
            return {k: self._resolve_arg(v, context) for k,v in arg.items()}
        if isinstance(arg, list):
            return [self._resolve_arg(v, context) for v in arg]
        return arg

    def run_plan(self, plan: dict):
        context = {}
        for step in plan.get("steps", []):
            tool_name = step["tool"]
            fn = self.tool_registry.get(tool_name)
            if fn is None:
                raise KeyError(f"Tool not found: {tool_name}")
            args = step.get("args", {})
            resolved = self._resolve_arg(args, context)
            self.log(f"[RUN] {step['id']} → {tool_name}({resolved})")
            out = fn(**resolved) if isinstance(resolved, dict) else fn(resolved)
            context[step["out"]] = out
        expect = plan.get("expect", {})
        result = {
            "artifacts": expect.get("artifacts", []),
            "metrics": self._resolve_arg(expect.get("metrics"), context),
            "report": self._resolve_arg(expect.get("report"), context),
        }
        return result
