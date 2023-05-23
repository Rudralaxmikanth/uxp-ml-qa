from transformers import pipeline

question_answers = pipeline(task='question-answering')

with open("documentation.doc", "r") as file:
    uxpdoc = file.read()


while True:
    My_input=input('Question:' )
    if My_input=='done':
        break
    preds = question_answers(
    question= My_input,
   
    context=uxpdoc
      )

    print(f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}")
