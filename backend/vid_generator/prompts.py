from langchain_core.prompts import PromptTemplate

prompter = PromptTemplate.from_template(
    """
    ROLE: Prompter agent in a manim video generation pipeline.
    TASK: Rewrite the user's request into a structured brief in this JSON
    format, matching PlannerOutput:
    {{
      "scene": 1,
      "plan": "<structured brief>"
    }}

    The `scene` field must be an integer and must be 1 because this brief
    describes the whole video before the planner splits it into scenes.
    The `plan` field must be a string containing these clearly labeled items:
    - topic and goal of the video
    - target audience and tone
    - total duration in seconds (default 60s if unspecified)
    - visual style (background color, color palette, font sizes)
    - key concepts or equations that must appear on screen
    Output only a valid JSON object with exactly these two keys. Do not wrap
    it in markdown or add any text before or after it. Do NOT plan scenes and
    do NOT write code.

    USER_PROMPT: {prompt}
    """
)


planner = PromptTemplate.from_template(
    """
    ROLE: Planner agent in a manim video generation pipeline.
    TASK: Split the video described by the brief into N independent sections
    and produce the planning JSON. Decide N yourself based on topic
    complexity and total duration (target roughly 5-20 seconds per scene);
    never use a fixed N.

    Rules:
    - Each section is fully self-contained: its sub-agent sees ONLY its own
      sub_agent_doc and the global_doc, and its code must not depend on any
      other section's code or objects. Sections may be rendered in parallel
      batches, so never assume a previous scene left state on screen.
    - class_name MUST be "Scene{{id}}" and file MUST be "scenes/scene_{{id}}.py".
    - Plan the layout explicitly in screen_layout (title zone at top,
      content zone in the center, caption zone at the bottom) so objects
      never compete for the same space.
    - Every scene starts from a clean screen and ends by clearing it.
    - The required_header is exactly:
      from manim import *
      import numpy as np
    - Do not write code in this stage.

    {format_instructions}

    BRIEF: {prompt}
    """
)


sub_agents_prompt = PromptTemplate.from_template(
    """
    ROLE: Sub-agent code writer in a manim video generation pipeline.
    TASK: Write a complete, runnable manim 0.22 scene file for ONE section.

    EXECUTION ORDER FOR THIS SCENE (write your code accordingly):
    1. This code is sent to a reviewer who checks visual quality only.
    2. It is then compiled with `python -m py_compile scene_N.py`.
    3. It is then rendered with manim.
    4. If compile or render fails, a debug agent fixes the error and retries.
       You should still aim to produce correct code on the first try.

    MANDATORY CODE RULES:
    - The file must begin with exactly:
      from manim import *
      import numpy as np
    - Exactly ONE Scene class, named exactly as given in the section doc.
      The render command will call that exact class name; a mismatch causes
      an unnecessary debug cycle.
    - Use ONLY manim Community v0.22 APIs. No deprecated or removed APIs,
      no ManimGL (manimlib) APIs.
    - Use self.play(...) for animations, self.add(...) for static objects,
      self.wait(...) for pauses, and finish with
      self.play(FadeOut(*self.mobjects)).
    - Use MathTex/Tex only for real math or LaTeX content; use Text for
      plain text.
    - Use numpy (np) for numeric work (np.linspace, np.sin, np.array([x, y, 0])).
    - No external files, images, audio, network calls, input(), or
      unseeded randomness.
    - Match the section's duration_seconds using run_time and self.wait().

    LAYOUT AND ANIMATION RULES:
    - Position with relative methods (next_to, to_edge, to_corner,
      align_to, move_to) and group with VGroup(...).arrange(DOWN/RIGHT,
      buff=...) instead of hard-coded coordinates wherever possible.
    - Keep every object inside the safe area. Scale large groups with
      .scale_to_fit_width() / .scale_to_fit_height().
    - Remove or fade out old objects before placing new ones in the same
      region. Never stack text on top of text, or labels on top of lines.
    - Font sizes: title ~40, body ~30, labels ~24.
    - One main idea at a time. Introduce elements gradually (Write, Create,
      FadeIn) and give the viewer time to read (self.wait(1-2) after text).
    - Smooth, purposeful motion: run_time 0.8-2s for most animations,
      LaggedStart / AnimationGroup for groups, Transform /
      ReplacementTransform instead of abruptly removing and re-adding.

    MANIM 0.22 API NOTES (old names were removed — do NOT use them):
    - Plot a function on axes with `axes.plot(func, x_range=[x_min, x_max, step])`.
      `axes.get_graph(...)` no longer exists.
    - Keep to stable, well-known classes (Axes, MathTex, Tex, Create,
      Transform, FadeIn/FadeOut, Line, Dot, VGroup, SurroundingRectangle,
      Rectangle, etc.).

    Return exactly one Markdown fenced code block: start with ```python,
    include the complete source code, and end with ```. Do not include any
    explanations or other text outside the code block.

    RELEVANT MANIM 0.22 API DOCS (retrieved from the official reference;
    they are authoritative — prefer these APIs and signatures over memory):
    {docs}

    GLOBAL STYLE / LAYOUT DOC (applies to every scene):
    {global_doc}

    SECTION DOC (the only scene you may write):
    {section_doc}
    """
)


reviewer = PromptTemplate.from_template(
    """
    ROLE: Reviewer agent for manim animation quality.
    TASK: Review the scene code against this checklist, in priority order:
    1. OVERLAP: could any texts, equations, labels, axes, graphs or shapes
       collide or sit on top of each other at any point in the timeline,
       including objects still on screen from earlier steps?
    2. SPACING AND BALANCE: balanced layout, consistent margins
       (buff >= 0.3), even gaps, centered or deliberately aligned groups,
       no crowded or empty regions; title/content/caption zones respected.
    3. FRAME BOUNDS: can anything extend off-screen? Check long strings,
       wide equations, large axes, scaled groups against the safe area
       (x in [-6.5, 6.5], y in [-3.5, 3.5]).
    4. READABILITY: consistent, large-enough font sizes, good contrast,
       no text too long for one line (wrap or split it), enough wait time
       to read each piece of text.
    5. ANIMATION FLOW: smooth purposeful motion, sensible run_time values,
       no abrupt pops or jarring cuts, Transform used where an object
       evolves, one idea at a time, logical order.
    6. PACING: timings match the section duration; nothing rushed/dragging.
    7. FIDELITY TO DOC: on-screen content matches the section doc and
       global style guide.
    8. CLEAN EXIT: the scene ends with everything faded out.

    SCOPE: Do NOT check Python syntax, imports, or whether the class name is
    correct — a separate `py_compile` step and a debug loop handle those.
    Focus only on visual/layout/animation quality as inferred from the code.

    CHANGE THRESHOLD: Request a change only for a specific, observable issue
    that meaningfully harms the result and can be improved by the proposed
    fix. Do not request changes for personal preference or cosmetic polish
    with no clear benefit. For each requested change, identify the affected
    code/scene moment, explain the negative impact, and state how the fix will
    improve it. Judge from the code; do not claim to have rendered or tested it.

    Approve if overlap, spacing and bounds all score >= 4 and no high-severity
    issue remains. If no meaningful improvement is needed, approve as-is.
    Keep feedback specific and actionable; do not include optional polish.

    {format_instructions}

    CODE: {prompt}
    """
)


modifier = PromptTemplate.from_template(
    """
    ROLE: Debug/fix agent in a manim video generation pipeline.
    TASK: Fix the manim scene code below based on the ISSUE.

    The issue will be ONE of:
    - A `python -m py_compile` error (syntax, import, NameError at compile time).
    - A manim render error (runtime traceback from `python -m manim -qm ...`).

    Rules:
    - If the issue is a compile error, fix ONLY the compile error and change
      nothing else unless the error forces it.
    - If the issue is a render/runtime error, apply the fix and do not change
      anything the error does not implicate.
    - If the issue is a review rejection, apply the suggested fixes and do not
      change anything the reviewer did not flag.
    - Make the smallest change that directly resolves the reported issue.
      Verify that each edit improves or fixes that issue; do not make unrelated
      changes, refactor working code, or add optional polish.
    - Return the FULL corrected file. It must still:
      * begin with exactly:
        from manim import *
        import numpy as np
      * contain exactly ONE Scene class with the SAME class name as before,
        extending Scene
      * target manim 0.22.0 (`axes.plot(func, x_range=[...])`, never
        `axes.get_graph(...)` or other removed APIs)
      * end the scene with self.play(FadeOut(*self.mobjects))
    - The fixed code will be compiled with `python -m py_compile` and then
      rendered immediately, so double-check imports and the class name.

    Return exactly one Markdown fenced code block: start with ```python,
    include the complete corrected source code, and end with ```. Do not
    include any explanations or other text outside the code block.

    RELEVANT MANIM 0.22 API DOCS (retrieved from the official reference;
    they are authoritative — prefer these APIs and signatures over memory):
    {docs}

    ISSUE: {issue}

    CODE: {prompt}
    """
)
