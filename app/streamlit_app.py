"""App de Streamlit: cuaderno de evaluación de comentarios estudiantiles.

Ejecutar desde la raíz del repositorio:  streamlit run app/streamlit_app.py
Usa src/modelo.py directamente (no necesita la API).
"""
import io
import sys
from html import escape
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import altair as alt
import pandas as pd
import streamlit as st

from src.modelo import CLASES, obtener_modelo, predecir, predecir_lote
from src.modelo import nota_confianza as detalle_modelo
from src.preprocesamiento import preparar

MAX_CARACTERES = 5000
COLORES = {"negativo": "#C4542D", "neutral": "#8A857C", "positivo": "#6B7B3A"}
TINTA, TINTA_SUAVE, RENGLON = "#1B2A41", "#47546A", "#C3CEDB"
NOMBRES_TEXTO = ("comentario", "comentarios", "texto", "text", "comment", "opinion", "opinión", "review")

st.set_page_config(page_title="Cuaderno de polaridad", page_icon=":material/edit_note:", layout="wide")


@st.cache_resource(show_spinner="Abriendo el cuaderno…")
def modelo():
    return obtener_modelo()


def pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def num(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def html(fragmento: str) -> None:
    st.html(fragmento)


html(f"<style>{(RAIZ / 'app' / 'estilos.css').read_text(encoding='utf-8')}</style>")

try:
    m = modelo()
except Exception as e:  # hash incorrecto, versión equivocada, archivo ausente…
    st.error(f"No se pudo cargar el modelo de forma segura: {e}")
    st.stop()

f1 = m.info.get("macro_f1_test")
html(f"""
<header class="cab">
  <div>
    <p class="sobretitulo">Sistemas Inteligentes · Proyecto</p>
    <h1 class="titulo">Polaridad de los comentarios <em>estudiantiles</em></h1>
    <p class="autores">Cuaderno de evaluación por <b>Emily Bonilla</b> y <b>Santiago Grande</b></p>
  </div>
  <aside class="nota-margen">
    <dl>
      <dt>modelo</dt><dd>{escape(m.info.get("modelo", "—"))}</dd>
      <dt>versión</dt><dd class="mono">{escape(m.version)}</dd>
      <dt>F1 macro</dt><dd class="mono">{f"{f1:.3f}".replace(".", ",") if f1 is not None else "—"} <span style="color:var(--tinta-suave)">(prueba)</span></dd>
      <dt>clases</dt><dd class="clases-leyenda">{"".join(f'<span class="punto punto--{c}">{c}</span>' for c in CLASES)}</dd>
    </dl>
  </aside>
</header>
<div class="renglones" aria-hidden="true"></div>
""")


def sello(resultado: dict | None) -> str:
    if resultado is None:
        return """<div class="sello sello--vacio"><span class="sello-etq">pendiente</span>
                  <span class="sello-clase">sin calificar</span></div>"""
    c = resultado["clase"]
    incierto = resultado.get("confianza_baja", False)
    etiqueta = "incierto" if incierto else "calificado"
    return f"""<div class="sello sello--{c}{' sello--incierto' if incierto else ''}" role="img"
      aria-label="Clase: {c}{', confianza baja' if incierto else ''}">
      <span class="sello-etq">{etiqueta}</span>
      <span class="sello-clase">{c}</span>
      <span class="sello-pie">confianza {pct(resultado["probabilidades"][c])}</span></div>"""


def barras(probabilidades: dict) -> str:
    maxima = max(probabilidades, key=probabilidades.get)
    filas = "".join(
        f"""<div class="prob prob--{c}{' prob--max' if c == maxima else ''}">
              <span class="prob-nombre">{c}</span>
              <span class="prob-trazo"><i style="width:{max(p * 100, 0.6):.2f}%"></i></span>
              <span class="prob-valor">{pct(p)}</span></div>"""
        for c, p in probabilidades.items()
    )
    return f'<div class="probs">{filas}</div>'


def nota_confianza(resultado: dict, detalle: str) -> str:
    if not resultado.get("confianza_baja"):
        return ""
    return f"""<aside class="nota-aviso" role="note">
      <p class="nota-aviso-titulo">Confianza baja</p>
      <p class="nota-aviso-texto">{escape(resultado["advertencia"])}</p>
      <p class="nota-aviso-cifra">{escape(detalle)}</p></aside>"""


tab_uno, tab_csv = st.tabs(["Un comentario", "Archivo CSV"])

# ----------------------------------------------------------------- un comentario
with tab_uno:
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        html("""<p class="seccion">Escribe el <em>comentario</em></p>
                <p class="ayuda">Tal como lo dejaría un estudiante sobre su curso. Correos, teléfonos y nombres
                de docentes se anonimizan antes de pasar por el modelo. Máximo 5.000 caracteres.</p>""")
        with st.form("form_uno", border=False):
            texto = st.text_area(
                "Comentario",
                height=224,
                max_chars=MAX_CARACTERES,
                placeholder="Ej.: El curso fue muy dinámico y la profesora explicaba con claridad…",
            )
            enviado = st.form_submit_button("Calificar comentario", type="primary")

        if enviado:
            if not texto.strip():
                html('<p class="anotacion"><b>Falta el texto.</b> Escribe un comentario antes de calificarlo.</p>')
                st.session_state.pop("resultado_uno", None)
            else:
                st.session_state["resultado_uno"] = {
                    "res": predecir(texto),
                    "leido": preparar(texto),
                    "detalle": detalle_modelo(texto),
                }

    with der:
        guardado = st.session_state.get("resultado_uno")
        html('<p class="rotulo">Calificación</p>')
        if guardado is None:
            html(f'<div class="hoja-resultado">{sello(None)}'
                 '<p class="ayuda">El sello aparecerá aquí con la probabilidad de cada clase.</p></div>')
        else:
            res = guardado["res"]
            aviso = nota_confianza(res, guardado["detalle"])
            html(f"""<div class="hoja-resultado">{sello(res)}{aviso}{barras(res["probabilidades"])}
                <div class="leido"><p class="rotulo">Así lo leyó el modelo</p>
                <p class="mono">{escape(guardado["leido"]) or "∅"}</p></div></div>""")

# ----------------------------------------------------------------- archivo CSV
def leer_csv(datos: bytes) -> pd.DataFrame:
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(datos), sep=None, engine="python", encoding=codificacion)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo como texto UTF-8 ni Latin-1.")


def grafico_resumen(resumen: pd.DataFrame) -> alt.Chart:
    resumen = resumen.assign(etiqueta=[f"{num(n)}  ·  {pct(p)}"
                                       for n, p in zip(resumen["conteo"], resumen["proporcion"])])
    base = alt.Chart(resumen).encode(
        y=alt.Y("clase:N", sort=list(CLASES), title=None,
                axis=alt.Axis(labelFont="IBM Plex Sans", labelFontSize=14, labelColor=TINTA,
                              ticks=False, domain=False, labelPadding=12)),
        x=alt.X("conteo:Q", title=None, axis=None,
                scale=alt.Scale(domain=[0, max(int(resumen["conteo"].max()), 1) * 1.45])),
    )
    barras_ = base.mark_bar(height=12, cornerRadiusEnd=2).encode(
        color=alt.Color("clase:N", scale=alt.Scale(domain=list(CLASES), range=[COLORES[c] for c in CLASES]),
                        legend=None),
        tooltip=[alt.Tooltip("clase:N", title="Clase"), alt.Tooltip("conteo:Q", title="Comentarios"),
                 alt.Tooltip("proporcion:Q", title="Proporción", format=".1%")],
    )
    texto = base.mark_text(align="left", dx=10, font="IBM Plex Mono", fontSize=13, color=TINTA).encode(
        text="etiqueta:N")
    guia = alt.Chart(resumen).mark_rule(color=RENGLON, strokeDash=[2, 3]).encode(
        y=alt.Y("clase:N", sort=list(CLASES), bandPosition=1))
    return (
        (guia + barras_ + texto)
        .properties(height=190)
        .configure(background="transparent", font="IBM Plex Sans")
        .configure_view(strokeWidth=0)
    )


with tab_csv:
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        html("""<p class="seccion">Clasifica un <em>archivo</em></p>
                <p class="ayuda">Sube un CSV con una columna de comentarios. El archivo se procesa en memoria y no
                se guarda en ningún lado.</p>""")
        archivo = st.file_uploader("Archivo CSV", type=["csv"])
    with der:
        html("""<p class="rotulo">Qué vas a obtener</p>
                <p class="ayuda">El mismo archivo con cinco columnas nuevas: <code>clase</code>,
                <code>prob_negativo</code>, <code>prob_neutral</code>, <code>prob_positivo</code> y
                <code>confianza_baja</code>.
                Las filas sin texto quedan marcadas como <code>sin_texto</code>.</p>""")

    if archivo is not None:
        try:
            df = leer_csv(archivo.getvalue())
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()

        if df.empty or not len(df.columns):
            html('<p class="anotacion"><b>Archivo vacío.</b> El CSV no tiene filas para clasificar.</p>')
            st.stop()

        texto_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        opciones = texto_cols or list(df.columns)
        sugerida = next((c for c in opciones if str(c).strip().lower() in NOMBRES_TEXTO), None)
        if sugerida is None:  # la columna con textos más largos en promedio
            sugerida = max(opciones, key=lambda c: df[c].astype(str).str.len().mean())

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            columna = st.selectbox("Columna con el texto", opciones, index=opciones.index(sugerida))
            st.caption(f"{num(len(df))} filas · {len(df.columns)} columnas")
            clasificar = st.button("Clasificar archivo", type="primary")

        clave = (archivo.file_id, columna)
        if clasificar:
            with st.spinner("Calificando comentarios…"):
                textos = df[columna].fillna("").astype(str)
                hay_texto = textos.str.strip() != ""
                salida = df.copy()
                salida["clase"] = "sin_texto"
                for c in CLASES:
                    salida[f"prob_{c}"] = float("nan")
                salida["confianza_baja"] = pd.Series(pd.NA, index=salida.index, dtype="boolean")
                resultados = predecir_lote(textos[hay_texto].tolist())
                if resultados:
                    salida.loc[hay_texto, "clase"] = [r["clase"] for r in resultados]
                    for c in CLASES:
                        salida.loc[hay_texto, f"prob_{c}"] = [r["probabilidades"][c] for r in resultados]
                    salida.loc[hay_texto, "confianza_baja"] = [r["confianza_baja"] for r in resultados]
            st.session_state["resultado_csv"] = {"clave": clave, "salida": salida, "nombre": archivo.name}

        guardado = st.session_state.get("resultado_csv")
        if guardado and guardado["clave"] == clave:
            salida = guardado["salida"]
            clasificadas = salida[salida["clase"] != "sin_texto"]
            conteo = clasificadas["clase"].value_counts().reindex(list(CLASES), fill_value=0)
            total = int(conteo.sum())
            resumen = pd.DataFrame({"clase": list(CLASES), "conteo": conteo.values,
                                    "proporcion": conteo.values / total if total else [0.0] * 3})

            html('<div class="renglones" aria-hidden="true"></div>')
            r1, r2 = st.columns([2, 3], gap="large")
            with r1:
                filas = "".join(
                    f'<tr><td><span class="punto punto--{c}">{c}</span></td>'
                    f'<td class="num">{num(n)}</td><td class="num">{pct(p)}</td></tr>'
                    for c, n, p in resumen.itertuples(index=False)
                )
                sin_texto = len(salida) - total
                inciertos = int(clasificadas["confianza_baja"].sum())
                html(f"""<p class="rotulo">Resumen</p>
                    <p class="cifra-grande">{num(total)}<span>comentarios calificados</span></p>
                    <table class="resumen" style="margin-top:1.2rem">
                      <thead><tr><th>clase</th><th class="num">conteo</th><th class="num">%</th></tr></thead>
                      <tbody>{filas}</tbody>
                      <tfoot><tr><td colspan="3">{num(sin_texto)} fila(s) sin texto no se clasificaron.</td></tr></tfoot>
                    </table>
                    <p class="resumen-aviso"><span class="mono">{num(inciertos)}</span> comentario(s) con
                    <b>confianza baja</b> ({pct(inciertos / total) if total else pct(0)}). Revísalos a mano
                    antes de sacar conclusiones.</p>""")
            with r2:
                html('<p class="rotulo">Distribución por clase</p>')
                st.altair_chart(grafico_resumen(resumen), width="stretch", theme=None)

            html('<p class="rotulo" style="margin-top:1.8rem">Vista previa · primeras 200 filas</p>')
            st.dataframe(
                salida[[columna, "clase", *(f"prob_{c}" for c in CLASES), "confianza_baja"]].head(200),
                hide_index=True,
                column_config={f"prob_{c}": st.column_config.NumberColumn(f"prob_{c}", format="%.3f")
                               for c in CLASES},
            )
            st.download_button(
                "Descargar CSV clasificado",
                data=salida.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{Path(guardado['nombre']).stem}_clasificado.csv",
                mime="text/csv",
            )
