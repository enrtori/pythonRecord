import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return(e/e.sum())

logits = ([5.1, 4.5, -1.2, 5.8, 1.0])
palabras = (["gato", "perro", "mesa", "felino", "animal"])

probs = softmax(logits)

for p,prob in zip(palabras,probs):
    status_bar = "█" * int(prob * 40)
    print(f"{p:8s}: {prob:.3f}  {status_bar}")