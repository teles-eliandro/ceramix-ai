# Glossário de indicadores — CERAMIX-AI

Referência rápida de **cada número** exibido pela aplicação e pelo relatório.
Escrito para leitura por quem formula e queima esmalte, não só por cientista de dados.

---

## 1. Cor

### ΔE (Delta E) — *diferença de cor percebida*

**O que é:** distância numérica entre duas cores no espaço CIELAB. Quanto menor, mais parecidas.

**Escala:** 0 = idênticas · ~100 = preto vs branco.

| ΔE | Leitura prática |
|---|---|
| 0 – 1 | imperceptível mesmo lado a lado |
| 1 – 2 | perceptível só com observação treinada |
| 2 – 5 | perceptível a olho nu, ainda "mesma cor" |
| 5 – 10 | claramente cores diferentes |
| 10 – 20 | cores distintas |
| **> 20** | **cores sem relação visual** |

> ⚠️ O CERAMIX-AI opera hoje na faixa **ΔE 30–45**. Isso significa: **a cor prevista NÃO corresponde à cor pedida**. É a medida honesta da limitação do dado (ver §5).

**Fórmula usada:** CIE76 — $ΔE = \sqrt{(ΔL)^2 + (Δa)^2 + (Δb)^2}$

---

### RGB / HEX — *cor prevista*

**O que é:** a cor que o modelo prevê que a receita produz após a queima, em RGB (0–255) e hexadecimal.

**Como ler:** `#966C45` é um marrom médio. Se você pediu dourado `#B8860B` e recebeu `#966C45`, a previsão **errou** — mas o *intervalo* (§2) te diz se o acerto era possível.

> 💡 RGB é a cor de **tela**. A cor final de um esmalte depende do forno, espessura da camada e do barro — nenhum modelo captura isso a partir só da química.

---

### LAB — *cor em coordenadas perceptuais*

**O que é:** a mesma cor em três eixos, alinhados com a percepção humana.

| Canal | Nome | Significado | Faixa |
|---|---|---|---|
| **L\*** | luminosidade | 0 = preto · 100 = branco | 0–100 |
| **a\*** | verde ↔ vermelho | negativo = verde · positivo = vermelho | ~−128…127 |
| **b\*** | azul ↔ amarelo | negativo = azul · positivo = amarelo | ~−128…127 |

**Exemplo:** `L=59.22, a=9.86, b=62.73` → cor clara-média, levemente avermelhada, **fortemente amarelada** = o nosso dourado `#B8860B`.

> LAB é o padrão de fato na indústria cerâmica (ISO/CIE 11664-6) porque a *mesma* diferença numérica corresponde à *mesma* diferença percebida, em qualquer região do espaço de cor. Em RGB não é assim.

---

## 2. Incerteza

### Intervalo 90% — *faixa de cores plausíveis*

**O que é:** o modelo não afirma uma cor exata — afirma um **intervalo** onde a cor real deve cair, com 90% de confiança. Exibido como dois triplos RGB:

```
77,37,-3  →  222,180,141
(limite inferior)  (limite superior)
```

**Como ler:** se o alvo está *dentro* dessa faixa, **o resultado está dentro do que o modelo declara saber**. Não é acerto — é compatibilidade com a incerteza medida.

> Este intervalo é **calibrado** (técnica: *conformal prediction*). Testado: exigimos 90% de cobertura e medimos **91,1% / 90,3% / 87,3%** (canais R/G/B). Ou seja, os intervalos são honestos — não são decorativos.

| Largura do canal | Valor |
|---|---|
| R | ±72.7 |
| G | ±71.5 |
| B | ±72.1 |

> ⚠️ Uma largura de **±72 numa escala de 0–255** é enorme. É a tradução numérica de "não sei o suficiente". Nenhuma apresentação esconde isso — e é por isso que o intervalo aparece em vez de um valor único.

---

### "Alvo no intervalo" (in range) — *o alvo era alcançável?*

| Símbolo | Significado |
|---|---|
| ✅ **in range** | o alvo cai dentro do intervalo de 90% — resultado compatível |
| ❌ **fora** | o alvo está fora do intervalo — o modelo declara que **não acerta** esta |

**Como usar:** trate `in range` como **condição necessária, não suficiente**. Não significa "esta receita dará essa cor" — significa "esta receita não está descartada".

---

### score — *nota de ranqueamento (para ordenação interna)*

**O que é:** número usado **só para ordenar** a lista. Não é uma qualidade absoluta.

$$score = ΔE + 0{,}6 \times \text{gap} - 6 \times [\text{alvo dentro}] - 5 \times [\text{textura bate}] - 4 \times [\text{transparência bate}]$$

Ou seja: penaliza erro de cor, penaliza distância até o intervalo, e **bonifica** acertos de textura/transparência. **Menor = melhor.**

> ⚠️ O score é uma conveniência de ordenação. **Não compare scores entre buscas diferentes** — só têm sentido dentro de uma mesma lista.

---

## 3. Superfície e acabamento

### Textura (surface) — *acabamento tátil/óptico do esmalte*

9 classes previstas pelo modelo, da mais brilhante à mais fosca:

| Classe | Leitura |
|---|---|
| **Glossy** | brilhante, reflexo espelhado |
| **Semi-glossy** | brilho alto, reflexo levemente difuso |
| **Satin** | acetinado |
| **Satin-matte** | entre acetinado e fosco |
| **Matte** | fosco |
| **Semi-matte** | fosco com leve brilho |
| **Smooth Matte** | fosco uniforme |
| **Dry Matte** | fosco seco |
| **Stony Matte** | fosco com aspecto pedra |

**Acurácia medida:** 54,6% (baseline aleatório: 45,4%) — ganho real de **+9,3 p.p.**

> ⚠️ 54,6% significa que **quase metade das previsões de textura não corresponde à receita real**. Use como indicação, não como garantia.

**match / — (na coluna MATCH):**

| Símbolo | Significado |
|---|---|
| **match** | a textura prevista coincide com a que você pediu |
| **—** | não coincide, ou você não pediu textura |

---

### Transparência — *passagem de luz pelo esmalte*

| Classe | Leitura |
|---|---|
| **Opaque** | opaco — não passa luz (típico com opacificante) |
| **Semi-opaque** | quase opaco |
| **Translucent** | translúcido — passa luz difusa |
| **Transparent** | transparente — vê o barro através |

**Acurácia medida:** 58,6% (baseline: 43,4%) — ganho de **+15,3 p.p.**

---

## 4. Receita e queima

### Composição química (óxidos, % peso)

**O que é:** a análise química da receita em óxidos, em porcentagem de massa. É **a linguagem universal** da cerâmica técnica — duas receitas com ingredientes diferentes mas mesma composição de óxidos tendem a se comportar igual.

**Como ler os óxidos:**

| Grupo | Óxidos | Papel |
|---|---|---|
| **Formadores de vidro** | SiO₂, B₂O₃ | esqueleto do vidro |
| **Anfóteros** | Al₂O₃ | rigidez, resistência a cristalização |
| **Fundentes** | Na₂O, K₂O, Li₂O, CaO, MgO, BaO, SrO, ZnO, PbO | baixam o ponto de fusão |
| **Opacificantes** | SnO₂, ZrO₂, TiO₂, CeO₂ | tornam opaco |
| **Corantes** | CoO, CuO, Cr₂O₃, Fe₂O₃, MnO, NiO, V₂O₅, Pr₂O₃, Nd₂O₃ | a cor |
| **Outros** | P₂O₅, LOI, F | efeitos específicos; LOI = perda ao fogo |

> 💡 **O corante é o que mais importa para cor** — mas seu efeito depende da matriz e do oxigênio do forno. É exatamente por isso que a mesma receita pode queimar diferente (§5).

---

### Ingredientes (matérias-primas, gramas)

**O que é:** a receita **como se pesa na bancada** — materiais crus e quantidades. É o que você leva para a balança.

**Como ler:** proporções relativas. Para escalar, multiplique tudo pelo mesmo fator. O total difere de 100 g porque inclui adições (corantes, opacificantes) além da base.

---

### UMF — *Fórmula Molecular Unitária* (Unity Molecular Formula)

**O que é:** a receita recalculada em **moléculas relativas**, normalizada para que os fundentes somem 1,0. É o padrão profissional de comparação de esmaltes.

**Por que importa:** duas receitas com ingredientes totalmente diferentes podem ter UMF quase idênticas — e queimar igual. A UMF revela a estrutura real por trás dos números.

**Como ler:**
- fundentes somam **1,0** por definição
- **SiO₂** costuma ficar entre 2 e 4 (define vitrificação)
- **Al₂O₃** entre 0,2 e 0,6 (dureza, resistência)
- relação SiO₂:Al₂O₃ controla brilho ↔ fosco

---

### Cone — *temperatura de queima (escala Orton)*

**O que é:** a temperatura de queima na escala padrão **Orton**, mais usada que graus porque descreve o *efeito térmico* (o "calor acumulado"), não só pico de temperatura.

| Cone | ≈ Temperatura | Uso típico |
|---|---|---|
| 04–02 | 1060–1100 °C | baixa (biscoito, terracota) |
| **06–04** | 1000–1060 °C | baixa fusão |
| **5–6** | 1180–1240 °C | **stoneware** (mais comum em estúdio) |
| 9–10 | 1260–1300 °C | alta (porcelana) |

> ⚠️ O cone aparece como **referência da receita original** — é a queima para a qual ela foi projetada. O CERAMIX-AI **não recomenda** queima; reporta a que a receita pressupõe.

---

### Atmosfera — *química do forno*

| Valor | Significado | Efeito na cor |
|---|---|---|
| **Oxidation** | oxigênio abundante | cores "limpas"; cobre/titânio dão tons claros |
| **Reduction** | oxigênio reduzido | cobre→vermelho/sangue; ferro→azuis e verdes; **muda tudo** |

> 💡 A mesma receita em oxidação vs redução pode dar cores **completamente diferentes**. É por isso que a atmosfera é reportada junto — sem ela, a previsão de cor perde sentido.

---

## 5. Por que os números são baixos — a limitação central

Este glossário só é honesto se incluir isto.

**Medido neste dataset:**

| Achado | Valor |
|---|---|
| Variância de cor **dentro** da mesma química | **82,7%** |
| Teto teórico para qualquer modelo só-química | **≈ 0,17 – 0,28** |
| Dispersão entre receitas quimicamente idênticas | **47,8 unidades sRGB** |

**Tradução:** receitas **quimicamente idênticas** queimam com cores **visivelmente diferentes**. A causa está fora da química: espessura da camada, curva do forno, atmosfera real, barro de suporte.

**Consequência:** o ΔE alto não é falha de ajuste — é **limite de informação**. R²=0,347 está *acima* do teto previsto para o dado disponível.

**O que isso significa na prática:** este sistema faz **triagem honesta** de receitas reais, não garante cor exata. Para R² alto seria necessário dado industrial instrumentado (espectrofotômetro + XRF + curva de forno registrada) — que não existe publicamente.

---

*Última atualização: 2026-10-05 · Dataset: GlazyBench (arXiv:2605.06641, MIT, 21.691 receitas reais)*
