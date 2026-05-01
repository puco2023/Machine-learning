# LangGraph Chat Assistant

Ovaj projekat koristi [LangGraph](https://python.langchain.com/docs/langgraph/) i OpenAI GPT-4o model za kreiranje jednostavnog konzolnog asistenta koji vodi konverzaciju sa korisnikom dok korisnik ne napiše `"end"`.

## Funkcionalnosti

-  Unos pitanja kroz konzolu (`input("You: ")`)
-  Vođenje istorije konverzacije (`history`)
-  Odgovaranje pomoću OpenAI modela (`gpt-4o`)
-  Petlja pitanja/odgovora dok korisnik ne prekine

## Zahtevi

- Python 3.8+
- `openai`
- `langgraph`
- `langchain-openai`
- `python-dotenv`

Instaliraj potrebne biblioteke:

```bash
pip install openai langgraph langchain-openai python-dotenv
