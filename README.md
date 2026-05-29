# Fase I - Filtro de Risco e Econometria

Entrega da Fase I do projeto **Desenvolvimento de Web App para Gestao Quantitativa**.

O aplicativo funciona como um primeiro filtro de ativos, combinando analise fundamentalista de risco estatistico com modelos econometricos:

- Analise fundamentalista: setor, valor de mercado, P/L, P/VP, ROE, margem liquida, dividend yield e divida/patrimonio.
- CAPM: estimacao de beta, alfa e premio de risco.
- Fama-French 3 fatores: decomposicao dos retornos em mercado, tamanho (SMB) e valor (HML).
- ARCH/GARCH: modelagem da volatilidade condicional.
- Ranking consolidado: classificacao dos ativos por risco relativo.

## Como executar

Crie um ambiente virtual, instale as dependencias e rode o Streamlit:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Como usar

1. Informe os tickers separados por virgula.
2. Escolha um benchmark, por exemplo `^GSPC` para S&P 500 ou `^BVSP` para Ibovespa.
3. Defina o periodo da analise e a taxa livre de risco anual.
4. Consulte as abas de analise fundamentalista, CAPM, Fama-French, ARCH/GARCH e ranking.

Para ativos brasileiros no Yahoo Finance, normalmente e necessario usar o sufixo `.SA`, como `PETR4.SA`, `VALE3.SA` e `ITUB4.SA`.

Os precos sao baixados do Yahoo Finance com a biblioteca `yfinance`. Se o Yahoo Finance nao responder no momento da apresentacao, envie um CSV de precos reais pela barra lateral.

Os indicadores fundamentalistas tambem sao consultados no Yahoo Finance. Caso algum ativo nao tenha dados fundamentalistas disponiveis, os modulos econometricos continuam funcionando com a serie de precos.

O CAPM nao exige muitos ativos, mas exige uma serie historica suficiente: cada ativo precisa ter pelo menos 30 retornos diarios alinhados ao benchmark. Para uma apresentacao melhor, recomenda-se analisar entre 5 e 10 ativos.

## Arquivo de precos reais

O CSV de precos deve ter uma coluna de data e uma coluna para cada ativo, incluindo o benchmark:

| date | AAPL | MSFT | NVDA | ^GSPC |
| --- | --- | --- | --- | --- |
| 2024-01-02 | 185.64 | 368.85 | 48.17 | 4742.83 |

O app nao usa dados simulados. Se o Yahoo Finance falhar e nenhum CSV real for enviado, a execucao para com uma mensagem de erro.

## Arquivo de fatores Fama-French

O app tenta buscar automaticamente os fatores reais na Kenneth French Data Library usando `pandas-datareader`. Se essa fonte nao responder, envie o CSV manualmente.

A aba de Fama-French aceita um CSV opcional com as colunas:

| Coluna | Descricao |
| --- | --- |
| `date` | Data da observacao |
| `mkt-rf` | Retorno excedente do mercado |
| `smb` | Fator tamanho |
| `hml` | Fator valor |
| `rf` | Taxa livre de risco diaria, opcional |

Os fatores podem estar em formato decimal (`0.01`) ou percentual (`1.0`).

## Entrega

Arquivos principais:

- `app.py`: codigo do Web App.
- `requirements.txt`: dependencias do projeto.
- `README.md`: instrucoes de execucao e uso.
