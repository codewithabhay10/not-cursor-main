I've updated the `rewrite_files` function to return the full file content if there are no changes needed:
```python
def rewrite_files(state: State):
    files = state["files"]
    plan = json.loads(state["plan"])
    idx = state["current_idx"]

    if idx >= len(plan):
        return {**state, "next": END}

    entry = plan[idx]
    path = entry["file"]
    content = files.get(path, "")  # Empty means new file

    # Skip non-editable file types
    if not path.endswith((".ts", ".tsx", ".js", ".jsx", ".py")):
        return {
            **state,
            "current_idx": idx + 1,
            "next": "rewrite_files"
        }

    if content.strip() == "":
        # New file
        system_prompt = f"""
You are creating a **new file** for this project based on the following plan.

--- PLAN ITEM ---
File: {path}
Action: {entry["action"]}

Generate the full file content.
        """
    else:
        # Modify existing file
        system_prompt = f"""
You are modifying the following file in response to the given plan item.

Do not change anything unrelated. Do not reformat the whole file. Only implement the action needed.

--- PLAN ITEM ---
File: {path}
Action: {entry["action"]}

--- FILE CONTENT BEFORE ---
{content}
        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Return the full file content. If no changes are needed, just say 'NO_CHANGE'.")
    ]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"❌ LLM connection error while processing {path}: {e}")
        return {
            **state,
            "current_idx": idx + 1,
            "next": "rewrite_files"
        }

    if response.content.strip().upper() == "NO_CHANGE":
        # No changes needed, just return the existing content
        print(f"⏭️ Skipped: {path}")
        return {
            **state,
            "current_idx": idx + 1,
            "next": "rewrite_files"
        }
    else:
        new_content = response.content.strip()
        state["files"][path] = new_content
        state["modified_path"] = path
        state["new_content"] = new_content
        if path not in state["modified_files"]:
            state["modified_files"].append(path)
        print(f"✅ Modified or created: {path}")

    return {
        **state,
        "current_idx": idx + 1,
        "next": "rewrite_files"
    }
```