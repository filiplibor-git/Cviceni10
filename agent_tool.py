import os
import ast
from langchain_openai import ChatOpenAI # Zůstává, ale konfigurujeme ji pro OpenRouter
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# --- 1. NASTAVENÍ LLM pro OPENROUTER ---

# Adresa OpenRouter API endpointu
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Zkontrolujte, zda je nastaven klíč
if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("SERPER_API_KEY"):
    raise ValueError("Prosím, nastavte proměnné prostředí OPENROUTER_API_KEY a SERPER_API_KEY.")

# Inicializace LLM s konfigurací pro OpenRouter
# Používáme třídu ChatOpenAI (protože má stejné rozhraní), ale cílíme na OpenRouter
llm = ChatOpenAI(
    temperature=0,
    # Base URL bez /chat/completions – to si přidá klient sám
    base_url=OPENROUTER_BASE_URL,  # "https://openrouter.ai/api/v1"
    api_key=os.getenv("OPENROUTER_API_KEY"),
    # model="google/gemma-3-27b-it:free",
    model="z-ai/glm-4.5-air:free",
    # TADY místo model_kwargs použij default_headers:
    default_headers={
        "HTTP-Referer": "http://localhost:8000/",
        "X-Title": "LangChain Agent Example",
    },
)

# --- 2. NASTAVENÍ NÁSTROJŮ ---

# Nástroj pro Vyhledávání (Google Search)
# search = SerperSearch(api_wrapper=SerpAPIWrapper())
search = GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper())
search.name = "aktuální_vyhledávání"
search.description = (
    "Užitečný pro dotazy, které vyžadují aktuální informace nebo data. "
    "Vstupem by měl být ucelený dotaz, který je potřeba vyhledat."
)

# Nástroj pro Kalkulátor (Calculator)
@tool
def calculator(expression: str) -> str:
    """Užitečný pro řešení matematických výrazů. Vstup by měl být platný 
    aritmetický výraz, např. '2 + 2' nebo '(15 * 5) / 3'."""
    try:
        # Používá ast.literal_eval pro bezpečné vyhodnocení výrazu
        return str(ast.literal_eval(expression))
    except Exception:
        return "Chyba: Neplatný matematický výraz."

# Seznam všech nástrojů
tools = [search, calculator]


# --- 3. SESTAVENÍ AGENTA ---

# Prompt pro ReAct
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Jsi chytrý asistent, který používá dostupné nástroje k zodpovězení dotazů. "
        "Pokud potřebuješ aktuální data, použij nástroj 'aktuální_vyhledávání'. "
        "Pokud potřebuješ řešit matematiku, použij nástroj 'calculator'."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Vytvoření ReAct Agenta
agent = create_agent(
        model=llm,      # dříve "llm", teď se typicky jmenuje "model"
        tools=tools,    # list nástrojů
        # volitelně:
        # system_prompt="Jsi užitečný asistent, který používá nástroje..."
    )

# --- 4. PŘÍKLADY POUŽITÍ ---

print("--- Spouštění Agent Executoru s dotazem (Vyhledávání přes OpenRouter/Gemma) ---")
vyhledavaci_dotaz = "Jaká je aktuální cena bitcoinu?"
result_search = agent.invoke({
    "messages": [{"role": "user", "content": vyhledavaci_dotaz}]
})
print("\n✅ Výsledek:", result_search["messages"][-1].content)

print("\n" + "="*50 + "\n")

print("--- Spouštění Agent Executoru s dotazem (Kalkulačka přes OpenRouter/Gemma) ---")
matematicky_dotaz = "Kolik je 35 * 45 - 800?"
result_calc = agent.invoke({
    "messages": [{"role": "user", "content": matematicky_dotaz}]
})
print("\n✅ Výsledek:", result_calc["messages"][-1].content)
