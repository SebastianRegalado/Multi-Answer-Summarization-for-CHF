from annotated_text import annotated_text, parameters

parameters.SHOW_LABEL_SEPARATOR = False

question = "## Why do I get headaches?"

question
annotated_text(
    ("It is most likely stress related.", "cause", "#ff0000"),
    ("I would suggest goint to the doctor ASAP.", "suggestion", "#3d85c6"),
    ("I used to have terrible headaches during my undergrad.", "experience", "#8e7cc3"),
    ("In the meantime, try doing some yoga.", "treatment", "#bf9000"),
    ("I hope you feel better soon!", "information", "#8fce00")
)