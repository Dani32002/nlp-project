# Estructura del artículo final

Guía de secciones para el artículo del proyecto: qué contar en cada una y de dónde sale (sección del notebook o de la Entrega 1). Lo marcado con **[PENDIENTE]** todavía no está implementado o corrido.

---

## Título
Corto y que diga la tarea, el modelo y el organismo. Ejemplo: *Detección y corrección de sustituciones en secuencias codificantes de E. coli con un modelo de lenguaje genómico ajustado con LoRA*.

## Resumen (150–250 palabras)
- **Problema:** errores de sustitución en NGS (0.1–1 %); los correctores actuales dependen de la cobertura.
- **Propuesta:** tratar el ADN como lenguaje y la corrección como ortografía; se ajusta GENERanno con LoRA y una cabeza de clasificación por nucleótido.
- **Datos:** CDS de E. coli con errores simulados (3 tasas, 2 perfiles); arqueas como prueba fuera de dominio.
- **Resultado principal:** ganancia y F1 frente a las líneas base, con los números finales. **[PENDIENTE]**
- **Conclusión:** una frase sobre si se cumplieron los criterios de éxito.

---

## 1. Introducción
- **Motivación:**
  - Errores de NGS, que varían entre corridas (Stoler & Nekrutenko).
  - Dependencia de la redundancia y la cobertura (Mitchell et al.).
  - Utilidad de corregir una sola secuencia sin cobertura.
- **Analogía NLP:** nucleótido = letra, codón = palabra, CDS = oración, error de secuenciación = error ortográfico (Soft-Masked BERT).
- **Pregunta de investigación:** ¿puede un modelo genómico preentrenado detectar y corregir sustituciones usando solo el contexto?
- **Contribuciones** (en viñetas):
  1. Formulación como clasificación por token con GENERanno + LoRA.
  2. Evaluación separada de detección y corrección, que Mitchell et al. no hacen.
  3. Comparación con líneas base clásicas: NoChange, frecuencia y n-gramas con canal ruidoso.
  4. Análisis por tasa de error, perfil de error, categoría biológica y dominio (arqueas).

## 2. Trabajo relacionado
Una o dos frases por trabajo, agrupadas por tema y no como lista de resúmenes. Cada párrafo debe cerrar con en qué se diferencia de nuestro proyecto. Base: sección 4 de la Entrega 1.
- **Correctores clásicos basados en k-mers** (Mitchell et al.) y **la perplejidad de un LM como juez de la corrección** (Athena).
- **Transformers para corrección:**
  - DeepConsensus: usa redundancia e indels.
  - VESTIGE: enmascara posiciones ya conocidas; no tiene que encontrar el error.
- **Modelos fundacionales genómicos:** GENERanno (tokenización por nucleótido, atención bidireccional, 8192 de contexto).
- **Corrección ortográfica en NLP:** Soft-Masked BERT (detección + corrección, errores sintéticos).

## 3. Datos
- **Fuente:** RefSeq. E. coli K-12 MG1655 (GCF_000005845.2) y H. volcanii DS2 (GCF_000025685.1).
- **Filtros:**
  - Criterios: pseudogenes, parciales, excepciones de traducción, caracteres no ACGT, longitud no múltiplo de 3, codones de inicio o fin inválidos, duplicados.
  - Tabla de descartados (Entrega 1, §3.2).
  - Número final de genes y nucleótidos de cada genoma.
- **EDA resumido:** solo lo que justifica decisiones posteriores (detalle completo en el apéndice o por referencia a la Entrega 1).
  - Contenido de GC de E. coli (~50 %) frente al de las arqueas (~65 %): justifica la prueba fuera de dominio.
  - Sesgo por posición en el codón: motiva el análisis por categorías.
  - Codones de inicio y parada dominantes.
  - Longitudes menores que 8192: no hace falta truncar.
- **Partición** (notebook §2):
  - Proteínas agrupadas con MMseqs2 (50 % de identidad, 80 % de cobertura) para que los parálogos queden juntos.
  - GroupShuffleSplit 80/10/10 con semilla 42.
  - Tabla con genes y nucleótidos por partición.
- **Arqueas:** solo para prueba; según sus autores, GENERanno no las vio en el preentrenamiento.

## 4. Simulación de errores
Notebook §3, función `create_errors`.
- **Solo sustituciones**, y la base nueva siempre distinta de la original (lección de Athena).
- **Tasas:** 0.1 %, 1 % y 5 %.
- **Perfil uniforme.**
- **Perfil realista:**
  - Probabilidad creciente a lo largo de lecturas simuladas de 150 nt (pendiente 3).
  - ×3 tras homopolímeros de 3 o más bases, con 50 % de probabilidad de repetir la base anterior.
  - Reescalado para que la media sea la tasa.
  - Justificación: Stoler & Nekrutenko y la lección de VESTIGE sobre no entrenar con corrupción uniforme.
- **Entrenamiento:** errores generados al vuelo en cada época (aumentación).
- **Validación y prueba:** errores fijos generados una vez (validación con 1 semilla, prueba con 5) para que todos los modelos se evalúen sobre las mismas secuencias.
- **Arqueas:** mismo esquema que la prueba. **[PENDIENTE: generar]**

## 5. Métodos

### 5.1 Líneas base
Notebook §5 y §7.2.
- **NoChange:** copia la entrada; ganancia = 0 por construcción. Verifica que el modelo no se limita a copiar.
- **FrequencyCorrector:** sabe dónde están los errores y elige la base nueva al azar entre las otras tres, con probabilidad proporcional a su frecuencia en el entrenamiento (sugerido por VESTIGE). Es una referencia solo para la corrección: su detección es perfecta por construcción, así que solo se reportan sus métricas de corrección.
- **N-gramas con interpolación lineal y canal ruidoso:**
  - Contexto bidireccional; los 4 candidatos se puntúan por posición.
  - Prior: log(1−e) para la base observada y log(e/3) para las otras.
  - Explicar por qué hizo falta el prior: sin él se sobrecorregía cerca del 60 % de las posiciones.
  - Búsqueda en malla sobre validación: n ∈ {4, 7, 10, 13} × 4 patrones de λ (uniforme, lineal, geométrico, inverso) × e ∈ {0.1, 0.2, 0.3, 0.4} = 64 combinaciones, eligiendo por ganancia media en las 6 configuraciones.
- **GENERanno sin ajuste (zero-shot):** el modelo preentrenado con su cabeza MLM original, sin LoRA ni cabeza nueva.
  - Lee los logits de los tokens A, C, G y T en cada posición y propone la base más probable.
  - Mide cuánto sabe ya el modelo sin entrenar para la tarea, y por lo tanto cuánto aporta el fine-tuning.
  - Va cargado en la sección 5 del notebook; se evalúa con `run_epoch`. **[PENDIENTE: evaluar]**

### 5.2 Modelo propuesto
Notebook §6.
- **Backbone:** GENERanno-prokaryote-0.5b-base.
  - 26 capas, dimensión 1280, 16 cabezas, atención bidireccional.
  - Tokenización por nucleótido, de modo que una sustitución afecta un solo token.
- **Cabeza de clasificación:** MLP (1280 → 256 → 128 → 4, GELU, dropout 0.1) que predice la base correcta en cada posición. La detección sale de comparar la predicción con la entrada.
- **LoRA:** sobre q/k/v/o_proj, con α = 2·rango; la cabeza es totalmente entrenable (`modules_to_save`).
- **Figura:** diagrama de la arquitectura (Entrega 1, §5.1).

### 5.3 Entrenamiento
Notebook §7.
- **Pérdida:** entropía cruzada ponderada, con peso `error_weight` en las posiciones con error, por el desbalance (99 % de las posiciones son correctas).
- **Optimización:**
  - AdamW, lr 2e-4 con warmup y decaimiento lineal.
  - Batch de 4, hasta 5 épocas, early stopping con paciencia 2.
  - Selección del modelo por ganancia en validación.
  - Gradient checkpointing.
  - bf16 o fp32 según la GPU.
- **Búsqueda escalonada** (11 modelos):
  1. Peso de error ∈ {1, 5, 20, 100} con rango 8 sobre 1 % uniforme.
  2. Rangos 4 y 16 con el mejor peso.
  3. La mejor combinación entrenada en las 6 configuraciones (3 tasas × 2 perfiles).
  - Justificar por qué es escalonada: costo de cómputo.
- **Hardware:** Colab (GPU, tiempo por época) y una tabla de hiperparámetros finales.

## 6. Protocolo de evaluación
- **Comparación posición por posición** entre original, ruidosa y corregida; no hace falta alinear.
- **Métricas** (dar fórmulas):
  - **Detección:** P, R y F1. TP = error cambiado, FP = posición correcta cambiada, FN = error sin cambiar.
  - **Corrección:** P = errores arreglados / cambios; R = errores arreglados / errores; F1.
  - **Ganancia** (métrica principal) = (errores arreglados − posiciones correctas modificadas) / errores. Vale 1 si es perfecta, 0 para NoChange y es negativa si daña más de lo que arregla.
  - **Sobrecorrección** = posiciones correctas modificadas / posiciones correctas.
  - **AUPRC** con puntaje 1 − p(base de entrada). Es independiente del umbral y el azar obtiene la tasa de error.
  - **% de genes completamente correctos** (nivel "oración"). **[PENDIENTE]**
  - Explicar por qué no se usa exactitud: NoChange obtiene 99.9 %.
- **Evaluación solo en las posiciones con error (límite superior, tipo ablación):**
  - Se le da al modelo la ubicación de los errores (detección perfecta); en cada una elige la base más probable entre las 3 distintas de la entrada.
  - Referencia del azar: 33 %. FrequencyCorrector corresponde a este mismo escenario con elección aleatoria ponderada.
  - Separa cuánto se pierde por no detectar y cuánto por no saber corregir (siguiendo a Soft-Masked BERT).
- **Evaluación a nivel de codón y por categorías:** recall de detección y exactitud de corrección desglosados por:
  - Posición del error en el codón (1.ª, 2.ª o 3.ª).
  - Efecto en la proteína: sinónimo, no sinónimo o sin sentido (crea un stop prematuro).
  - Tipo de sustitución (matriz 4×4, A→C, A→G, …).
  - Ubicación: codones de inicio y parada frente al interior del gen.
- **Prueba:** 5 semillas de errores; media ± desviación estándar.
- **IC 95 %** por bootstrap remuestreando genes completos, porque las posiciones de un gen no son independientes. **[PENDIENTE]**
- **Criterios de éxito** (definidos antes de ver resultados):
  1. Ganancia mayor que 0 en todas las tasas.
  2. Superar al n-grama en F1 de detección y de corrección con un IC de la diferencia que excluya el 0.
  3. Ganancia mayor que 0 en arqueas.

## 7. Resultados
**[PENDIENTE]** Solo hechos y números; la interpretación va en la sección 8.
- **7.1 Selección de hiperparámetros:**
  - Tabla del peso de error y del rango con la ganancia en validación.
  - Mejores parámetros del n-grama, más la tabla top-5 de la malla.
  - Curvas de pérdida y de ganancia en validación (`plot_loss`, `plot_val_metrics`).
- **7.2 Resultado principal en prueba (E. coli):**
  - Tabla de modelos × métricas para cada tasa y perfil, con media ± desviación estándar e IC.
  - Modelos: NoChange, N-grama, GENERanno zero-shot y GENERanno + LoRA.
  - Frecuencia va aparte, solo con las métricas de corrección.
- **7.3 Matriz tasa de entrenamiento × tasa de prueba:** heatmap de ganancia. ¿Sobrecorrige el modelo entrenado al 5 % cuando la tasa real es 0.1 %?
- **7.4 Perfil uniforme frente a realista:** cruzado entrenamiento/prueba si se evalúa así.
- **7.5 Límite superior de corrección:** exactitud de cada modelo evaluado solo en las posiciones con error (definido en §6), contra el 33 % del azar y FrequencyCorrector. **[PENDIENTE]**
- **7.6 Fuera de dominio (arqueas):**
  - Caída de cada modelo respecto a E. coli.
  - Relación con el contenido de GC por gen (dispersión).
- **7.7 Verificación de los criterios de éxito:** tabla de sí/no.

## 8. Análisis y discusión
**[PENDIENTE]**
- **Por categorías** (definidas en §6), con lo que se espera de cada una:
  - Posición en el codón: la 3.ª debería ser la más difícil, porque concentra las sustituciones sinónimas.
  - Efecto en la proteína: los errores sin sentido deberían ser los más fáciles de detectar.
  - Tipo de sustitución: comparar con los sesgos reportados por Stoler & Nekrutenko.
  - Inicio y parada: deberían ser fáciles, porque esos codones están dominados por pocos valores.
- **Errores del modelo:** muestra de falsos positivos. ¿"Corrige" codones raros pero válidos hacia codones frecuentes? ¿Distingue error de variación natural? Conecta con la motivación.
- **Interpretación:**
  - Por qué gana o pierde frente al n-grama.
  - Papel del peso de error en el balance entre detección y sobrecorrección.
  - Posible memorización de E. coli en el preentrenamiento, contrastada con las arqueas.

## 9. Limitaciones y trabajo futuro
- **Limitaciones:**
  - Errores simulados sobre CDS, no lecturas reales; no es comparable directamente con los correctores de Mitchell et al.
  - Solo sustituciones, sin indels.
  - Búsqueda de hiperparámetros escalonada y limitada por el cómputo; una sola semilla de entrenamiento.
  - Un solo organismo de entrenamiento y uno fuera de dominio.
- **Trabajo futuro:**
  - Lecturas reales con gold standard.
  - Indels, con una pérdida basada en alineamiento como DeepConsensus.
  - Soft-masking con una red de detección explícita.
  - Más organismos.

## 10. Conclusiones
Un párrafo: respuesta a la pregunta de investigación, el resultado clave con su número y qué implica.

## Referencias
Las de la Entrega 1 (IEEE), más MMseqs2, LoRA (Hu et al., 2021) y PEFT si se citan.

## Apéndices (opcional)
- EDA completo (gráficas de la Entrega 1).
- Tabla completa de la malla de n-gramas.
- Historiales de entrenamiento de los 11 modelos.
- Detalles de reproducibilidad: versiones (transformers 4.57.3, peft 0.13.2), revisión del modelo `d02db0f`, semillas y hardware.
