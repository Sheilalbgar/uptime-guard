# Memoria · Uptime Guard

**Predicción de fallos de discos duros a 30 días en un centro de datos**

Sheila Albarrán García · Proyecto de Machine Learning · noviembre de 2026

---

## 1. El problema

Un centro de datos funciona con miles de discos duros, y cada mes algunos se estropean. Si un disco falla por sorpresa, hay que recuperar datos, mover cargas de trabajo y, en el peor caso, se para un servicio. Según el *Annual Outage Analysis 2026* del Uptime Institute, el 57 % de los operadores dice que su última caída importante costó más de 100.000 $, y uno de cada cinco habla de más de un millón.

La pregunta del proyecto es sencilla: **¿qué discos van a fallar en los próximos 30 días?** Si el equipo de operaciones lo sabe con antelación, puede cambiarlos de forma planificada, que es mucho más barato que hacerlo tras una avería.

- **Variable objetivo:** `falla_30d` (1 = el disco falla en los próximos 30 días; 0 = no).
- **Tipo de problema:** clasificación binaria con clases desbalanceadas.
- **Resultado que entrego:** una probabilidad de fallo por disco y un semáforo de acción: 🔴 sustituir, 🟠 vigilar y 🟢 OK.

## 2. Elección de la métrica

El error más caro es el **falso negativo**: el modelo dice «sano» y el disco falla. El falso positivo, revisar o cambiar un disco que estaba bien, cuesta bastante menos. Por eso me fijo sobre todo en el **recall** de la clase 1, sin perder de vista la **precisión** para no saturar al equipo con falsas alarmas.

La accuracy no sirve: decir siempre «sano» acierta el 89 % y no detecta ni un fallo. Para comparar modelos uso la **PR-AUC** (`average_precision`), que resume precisión y recall sin depender del umbral y es la recomendada cuando la clase positiva es rara.

Como las dos acciones tienen costes distintos, el modelo tiene **dos umbrales**. **Vigilar** es barato, así que ahí prima el recall. **Sustituir** cuesta dinero, así que ahí prima la precisión.

## 3. Datos y unión de fuentes

Los datos son de **Backblaze Drive Stats**, que publica cada trimestre el estado diario de unos 340.000 discos de sus centros de datos, con sus atributos SMART (contadores internos de errores, horas de uso, temperatura…).

Elegí 9 fechas de referencia, una por semana entre el 1 de enero y el 26 de febrero de 2026. No hay fechas en marzo porque cada fecha necesita 30 días completos por delante para saber si el disco falla: la del 26 de febrero mira hasta el 28 de marzo, el último margen que cabe en el trimestre. Así, los fallos de marzo sí se usan, como target de las fechas de febrero.

Para cada fecha uno tres fuentes por número de serie y fecha:

1. **La foto del día:** el estado de cada disco.
2. **La tabla de fallos del trimestre**, de la que sale el target: si el disco falla en los 30 días siguientes, vale 1.
3. **Las lecturas del mismo disco 7, 14 y 30 días antes**, para medir si empeora. Diciembre de 2025 aporta este historial para las primeras fechas.

Me quedo con todos los discos que fallan en los 30 días siguientes y con una muestra al azar de sanos (semilla 42), hasta llegar a **25.993 filas y 123 columnas**: 2.764 filas que fallan, de 902 discos distintos. Aparte guardo la foto del **1 de abril de 2026** (4.411 discos, ninguno de la muestra de enero y febrero) como prueba final.

Los datos no contienen información personal: solo números de serie y lecturas técnicas de los discos. Backblaze permite usarlos citando la fuente.

## 4. Preparación y análisis exploratorio

**Lo primero fue separar train y test, antes de mirar nada.** Un mismo disco puede aparecer en varias fechas, así que reparto discos y no filas: el 80 % a train y el 20 % a test, con la misma proporción de discos que fallan. Ningún disco está en los dos conjuntos.

Todas las decisiones de limpieza las tomé **solo con train**:

- **Missings:** cada fabricante mide los atributos SMART que quiere. Elimino los que tienen más del 50 % de huecos en train (183, 184, 187, 188, 189, 190, 241 y 242). Por ejemplo, el 187 y el 188 solo los mide Seagate; rellenarlos sería inventar datos de marcas que no los miden.
- **Feature engineering:** fabricante (sacado del modelo del disco), capacidad en TB, «disco nuevo» (sin lectura de hace 30 días) y **31 tendencias**: cuánto ha subido cada contador en 7, 14 y 30 días. Un contador de errores que crece dice más que su valor del día.
- A test y a abril les aplico exactamente las mismas transformaciones, sin aprender nada de sus datos.

Lo más relevante del análisis exploratorio:

- **El target está desbalanceado:** 89 % de sanos y 11 % de fallos. En la flota real fallan unos 0,9 de cada 1.000 discos al mes; la muestra tiene más fallos a propósito para que el modelo aprenda.
- **Los contadores de sectores dañados separan muchísimo** (Mann-Whitney, p < 0,001). El 51 % de los discos que fallan tiene sectores pendientes, frente al 2 % de los sanos. Las horas de uso también son significativas, pero separan poco: que algo sea significativo no quiere decir que sea importante para predecir.
- **Pearson y Spearman cuentan cosas distintas** con estos contadores tan asimétricos. Los atributos 192 y 193 dan 0,999 con Pearson y 0,11 con Spearman: el primero se deja arrastrar por unos pocos valores extremos. Ningún par de variables pasa de 0,9 con Spearman.
- **El fabricante está relacionado con el fallo** (chi², p < 0,001): HGST falla un 20 % y WDC un 6,5 %. Pero correlación no es causalidad: los HGST son los discos más viejos de la flota.

Preparé tres conjuntos de variables para compararlos: todas (46), sin las poco útiles (28) y un conjunto reducido sin redundantes (15).

## 5. Modelado

### Preprocesado

Todo va dentro de un `Pipeline` con `ColumnTransformer`: mediana para los huecos, log(1 + x) en las variables muy asimétricas, escalado solo para los modelos que lo necesitan (regresión logística, KNN y SVM) y One-Hot para el fabricante y el centro de datos, que son categóricas sin orden. Al estar dentro del pipeline, la mediana y el escalado se calculan solo con la parte de entrenamiento de cada pliegue.

### Validación

Uso `StratifiedGroupKFold` con 5 pliegues: cada pliegue mantiene la proporción de fallos y **nunca reparte las filas de un mismo disco entre dos pliegues**. Si no, el modelo se examinaría con discos que ya conoce.

### Referencias de partida (baseline)

- Decir siempre «sano»: 89 % de accuracy y 0 fallos detectados.
- **La regla de un técnico** (avisar si el disco tiene algún sector dañado): detecta el 66 % de los fallos con un 57 % de avisos acertados. Esa es la marca que hay que superar.

### Comparativa de modelos

Probé ocho modelos supervisados y uno combinado con PCA, todos con los mismos pliegues:

| Modelo | PR-AUC (CV) |
|---|---|
| LightGBM | 0,706 |
| **Random Forest** | **0,704** |
| CatBoost | 0,696 |
| PCA + Random Forest | 0,693 |
| Regresión logística L1 | 0,676 |
| XGBoost | 0,657 |
| KNN | 0,643 |
| Árbol de decisión | 0,630 |
| SVM | 0,545 |

La variación entre pliegues (±0,04) es mayor que las diferencias entre los primeros, pero mide sobre todo que unos pliegues son más difíciles que otros para todos los modelos. Por eso los comparé **pliegue a pliegue con un t-test pareado**. Random Forest supera a la regresión logística en 4 de 5 pliegues (p = 0,04) y empata con LightGBM (p = 0,31).

**Elegí Random Forest:** rinde igual que el mejor, tiene mejor precisión, pocos hiperparámetros y es más fácil de explicar que un boosting. Con él, los tres conjuntos de variables quedan a menos de 0,01 entre sí, así que me quedo con el **reducido (15 variables)**.

### Ajuste de hiperparámetros

`GridSearchCV` con tres métricas a la vez (PR-AUC, recall y precisión), `refit` por PR-AUC y nota de entrenamiento para vigilar el sobreajuste. Gana el modelo con **300 árboles, profundidad máxima 12, al menos 5 discos por hoja y sin pesos de clase**: PR-AUC 0,725 en validación cruzada (0,85 en entrenamiento). Sin límite de profundidad, el entrenamiento sube a 0,92 y la validación no mejora.

### Desbalanceo

Comparé cuatro opciones: no hacer nada, `class_weight`, SMOTE y undersampling (las dos últimas dentro del pipeline de imblearn). Las tres técnicas suben el recall con el umbral 0,5, pero **bajan la PR-AUC**: ordenan peor los discos por riesgo. Por eso no balanceo los datos y trato el desbalanceo con el umbral de decisión.

### ¿Hacían falta más filas?

La curva de aprendizaje sube de 0,692 con 3.300 filas a 0,726 con 16.600. Más filas sí ayudan, y por eso pasé de una primera versión de 5.000 filas a unas 26.000.

### No supervisado

- **PCA:** hacen falta 11 componentes de 26 para explicar el 90 % de la varianza. En dos dimensiones, los discos que fallan se concentran en una zona aunque el PCA no sabe cuáles son.
- **KMeans:** la silueta más alta es la de k = 2, pero me quedo con k = 4 porque los grupos se interpretan mejor. Aparece un grupo de 786 discos con muchos sectores pendientes en el que falla el 91 %. KMeans encuentra el deterioro sin ver el target.

## 6. Umbrales y evaluación

Los umbrales los elegí con predicciones de validación cruzada, sin tocar el test:

- **Vigilar (0,20):** el umbral más alto que todavía detecta al menos el 70 % de los fallos.
- **Sustituir (0,55):** el más bajo con al menos un 85 % de alarmas reales.

Después evalué **una sola vez en test** (5.163 filas de discos que el modelo no había visto):

| | Regla del técnico | Uptime Guard (test) | Uptime Guard (abril) |
|---|---|---|---|
| Fallos detectados, umbral vigilar | 66 % | **71 %** (IC 95 %: 67-74 %) | **76 %** |
| Alarmas reales, umbral sustituir | 57 % | **86 %** (IC 95 %: 82-90 %) | **86 %** |
| PR-AUC | — | 0,725 | 0,763 |

La PR-AUC en test coincide con la de validación cruzada, así que la validación agrupada estimaba bien. Los intervalos son de bootstrap (1.000 remuestreos), e incluso en su peor extremo el modelo supera a la regla del técnico. En abril, un mes posterior con discos nuevos, el resultado es igual o mejor: **el modelo generaliza al futuro**.

**¿En qué se fija?** Según la permutation importance en test, lo que más pesa son los sectores pendientes (SMART 197), cuánto han subido los sectores reasignados en el último mes y su valor actual (SMART 5). Tiene sentido físico: la superficie del disco se va dañando antes del fallo. Que una variable sea importante no la convierte en causa.

**¿Dónde falla?** El 89 % de los fallos que se escapan no tenía ninguna señal en los contadores de sectores: el disco murió de golpe. Ningún modelo con estos datos puede verlos; para eso están el RAID y las copias de seguridad. Por fabricante, acierta más en HGST (91 %) y menos en Toshiba (62 %).

## 7. Valor para el negocio

Con costes supuestos (5.000 € por fallo sorpresa, 300 € por cambio planificado y 50 € por falsa alarma) y la proporción real de fallos, esto es lo que ocurre en un mes en una flota como la de Backblaze, con 340.000 discos:

| Paso | En un mes | Dinero |
|---|---|---|
| Discos que van a fallar | unos **306** | — |
| Fallos que el modelo anticipa (55 %) | unos **168** | — |
| Coste evitado: cada uno se cambia por 300 € en vez de costar 5.000 € | 168 × 4.700 € | **+ 789.000 €** |
| Falsas alarmas: discos sanos que se revisan sin necesidad | unos **3.370** | 3.370 × 50 € = **− 169.000 €** |
| **Ahorro neto** | | **≈ 620.000 € al mes · 7,4 M€ al año** |

¿Y si un fallo por sorpresa cuesta más o menos? El ahorro cambia mucho:

| Si un fallo sorpresa cuesta… | Ahorro neto al año (340.000 discos) |
|---|---|
| 1.000 € | ❌ pérdidas (−0,6 M€): las falsas alarmas cuestan más de lo que se ahorra |
| ~1.300 € | ⚖️ ni se ahorra ni se pierde |
| 2.000 € | ✅ 1,4 M€ |
| **5.000 € (supuesto)** | ✅ **7,4 M€** |
| 10.000 € | ✅ 17,5 M€ |
| 20.000 € | ✅ 37,6 M€ |

El ahorro no está en el hardware (un disco nuevo cuesta unos 300 €), sino en evitar la avería por sorpresa: recuperar datos, horas de técnicos y servicio degradado. Un solo disco rara vez para el servicio, porque hay redundancia, pero si falla otro mientras se reconstruye el primero puede haber pérdida de datos o una caída. Según el Uptime Institute, el 57 % de las caídas importantes cuesta más de 100.000 $. Anticipar los fallos reduce ese riesgo.

Compensa a partir de unos 1.300 € por fallo, y cuanto más caro es un fallo, más rentable resulta. Como el coste real cambia de una empresa a otra, los umbrales están en `models/config_modelo.json` y se pueden ajustar sin reentrenar.

La **app de Streamlit** acerca el modelo a un usuario no técnico. Se sube un CSV con varios discos y devuelve el semáforo de cada uno, o se revisa un disco concreto con un formulario.

## 8. Limitaciones y próximos pasos

**Limitaciones**

- Los fallos repentinos, sin señal previa, no se pueden predecir con datos SMART.
- La muestra tiene un 11 % de fallos y la flota real un 0,09 %. En producción habría más falsas alarmas por cada acierto; por eso el ROI se calcula con la proporción real.
- Los costes del ROI son supuestos, no datos de Backblaze.
- El periodo es corto (enero y febrero para entrenar, abril para la prueba final), y el rendimiento cambia según el fabricante.
- La lista de variables asimétricas y el conjunto reducido se eligieron con todo train y no dentro de cada pliegue. El efecto es mínimo, pero lo dejo anotado.

**Próximos pasos**

- Probar en mayo y junio de 2026 para seguir el rendimiento mes a mes.
- Añadir variables de temperatura y carga de trabajo, si el centro de datos las tiene.
- Calibrar las probabilidades con la proporción real de fallos.
- Ajustar los umbrales con los costes reales de la empresa y reentrenar cada trimestre con los datos nuevos que publica Backblaze.

## 9. Referencias

- Backblaze. *Hard Drive Data and Stats* (Drive Stats). https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
- Uptime Institute (2026). *Annual Outage Analysis 2026*. https://intelligence.uptimeinstitute.com/resource/annual-outage-analysis-2026
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830. Documentación: https://scikit-learn.org (3.4 *Metrics and scoring*).
- Lemaître, G., Nogueira, F. y Aridas, C. K. (2017). *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning*. Journal of Machine Learning Research, 18(17), 1-5.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Streamlit. Documentación: https://docs.streamlit.io
- Funciones de visualización del curso (`bootcampviztools.py`).
