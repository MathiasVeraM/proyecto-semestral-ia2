"""Generador reproducible del dataset sintético de observaciones (español).

Combina (a) frases escritas a mano (incluye casos ambiguos, negaciones y lenguaje
informal) y (b) frases generadas por plantillas. Luego inyecta "ruido" realista
(mayúsculas, tildes faltantes, abreviaturas de chat, emojis, typos, espacios,
prefijos de turno, filas vacías y duplicados) para que el preprocesamiento sea
necesario.

Uso:  python -m src.generate_dataset
Salida: data/raw/observaciones_raw.csv  (columnas: id, texto, etiqueta)
"""
import random
import re
import unicodedata

import pandas as pd

from src.config import DATA_RAW, SEED

# ----------------------------------------------------------------------------
# Animales: (nombre, especie con artículo, terminación de género)
# ----------------------------------------------------------------------------
ANIMALES = [
    ("Max", "el perro", "o"), ("Luna", "la gata", "a"), ("Rocky", "el perro", "o"),
    ("Nala", "la perra", "a"), ("Simba", "el gato", "o"), ("Coco", "el loro", "o"),
    ("Bruno", "el perro", "o"), ("Mía", "la gata", "a"), ("Toby", "el perro", "o"),
    ("Kira", "la perra", "a"), ("Thor", "el perro", "o"), ("Canela", "la perra", "a"),
    ("Pelusa", "la coneja", "a"), ("Rufo", "el conejo", "o"), ("Estrella", "la yegua", "a"),
    ("Trueno", "el caballo", "o"), ("Manchas", "el gato", "o"), ("Lola", "la perra", "a"),
    ("Zeus", "el perro", "o"), ("Frida", "la gata", "a"), ("Chispa", "la perra", "a"),
    ("Pancho", "el gato", "o"), ("Rita", "la tortuga", "a"), ("Gaspar", "el mono", "o"),
    ("Dulce", "la cobaya", "a"), ("Tango", "el perro", "o"), ("Sol", "la gata", "a"),
    ("Apolo", "el perro", "o"), ("Maya", "la perra", "a"), ("Bola", "el gato", "o"),
    ("Panchita", "la cabra", "a"), ("Moka", "la perra", "a"), ("Jack", "el perro", "o"),
    ("Pepa", "la lora", "a"), ("Duque", "el caballo", "o"), ("Nube", "la gata", "a"),
]

APERTURAS = [
    "{Esp} {n}", "{n}", "Hoy {n}", "Durante la actividad {n}", "En el paseo {n}",
    "Tras la sesión de enriquecimiento {n}", "Al revisar a {n}", "{n} ({esp})",
    "Con {n}, {esp},", "Sobre {n}:", "Ya en el recinto {n}",
]

# ----------------------------------------------------------------------------
# Frases por clase. {o} = terminación a/o.
# ----------------------------------------------------------------------------
CLASE_0 = [
    "comió todo su alimento y estuvo muy activ{o}",
    "participó con entusiasmo en la actividad, buen apetito",
    "se mostró tranquil{o} y con ganas de jugar",
    "no presenta cojera ni heridas, camina normal",
    "sin vómitos ni diarrea, heces normales",
    "bebió agua con normalidad y descansó bien",
    "respondió bien al llamado y buscó interacción",
    "peso estable, pelaje brillante y ojos limpios",
    "tomó su medicación sin problema y sigue comiendo bien",
    "se adaptó bien al recinto nuevo, explora con curiosidad",
    "se dejó revisar sin estrés, todo en orden",
    "actividad sin novedades, comportamiento habitual",
    "mejoró mucho desde ayer, ya come normal",
    "jugó con la pelota un buen rato y se durmió relajad{o}",
    "no hubo vómitos, no hay sangrado y respira normal",
    "descartamos convulsiones, todo tranquilo durante el turno",
    "ya no tiene dificultad para respirar, evolución favorable",
    "no se observa decaimiento, está alert{o} y curios{o}",
    "ningún signo de dolor al tocar el abdomen, se ve contento",
    "se alimentó sin ayuda y no mostró molestias",
    "caminó todo el paseo sin cojear, buen ánimo",
    "interactuó con los voluntarios, cola en alto y apetito normal",
    "ladró poco, durmió tranquil{o} y comió su ración completa",
    "heridas cicatrizando bien, sin secreción ni mal olor",
    "buen día, nada que reportar",
    "recibió su vacuna y no tuvo ninguna reacción",
    "excelente actitud en la caminata, tomó agua y comió",
    "se bañó sin problemas y quedó muy relajad{o}",
]

CLASE_1 = [
    "comió solo la mitad de su ración",
    "estuvo algo decaíd{o} hoy pero aceptó caricias",
    "estornudó varias veces, sin otros síntomas",
    "cojea ligeramente de la pata trasera, pero camina",
    "una deposición blanda, hay que vigilar mañana",
    "estuvo más apátic{o} de lo habitual",
    "se rascó bastante, enrojecimiento leve en la oreja",
    "bebió menos agua de lo normal",
    "se mostró nervios{o} y evitó el contacto con otros animales",
    "tos ocasional, habrá que observar",
    "tiene una herida superficial en la pata, ya fue limpiada",
    "vomitó una vez después de comer, luego se vio normal",
    "no quiso jugar mucho, se cansó rápido",
    "pérdida de pelo localizada, revisar con el veterinario esta semana",
    "ojos con un poco de legaña, sin hinchazón",
    "come pero con poco apetito, está medio rar{o}",
    "algo inquiet{o} por la noche, no durmió bien",
    "apenas probó el snack, normalmente se lo come enseguida",
    "heces algo más oscuras de lo normal, sin otros cambios",
    "se lamió mucho una pata, no se ve herida pero conviene mirar",
    "respira un poco agitad{o} tras correr, se recuperó en minutos",
    "está más callad{o} que siempre, pero responde cuando lo llaman",
    "perdió algo de interés en el juego, sigue comiendo",
    "tiene la nariz seca y un leve decaimiento",
    "se negó a la caminata hoy, quizá por el calor",
    "rechazó una de las dos comidas",
]

CLASE_2 = [
    "no ha comido ni bebido desde ayer, está muy débil",
    "vómitos repetidos y no puede mantenerse en pie",
    "convulsiones durante la actividad, se llamó al veterinario de urgencia",
    "sangrado abundante en la herida, no se detiene",
    "dificultad para respirar, respiración muy agitada",
    "no responde a estímulos, está echad{o} sin moverse",
    "diarrea con sangre desde la mañana y está deshidratad{o}",
    "se desplomó al caminar y tiene las encías pálidas",
    "se negó a comer por tercer día consecutivo y pierde peso rápido",
    "agresión grave con otro animal, herida profunda y abierta",
    "hinchazón enorme en el abdomen, llora de dolor al tocar la zona",
    "no puede levantarse, arrastra las patas traseras",
    "posible intoxicación, babea mucho y tiembla sin parar",
    "temperatura muy alta y respira con la boca abierta, jadeo constante",
    "se ve fatal, no se para y no quiere ni agua",
    "tiene un golpe fuerte en la cabeza y está desorientad{o}",
    "ojo muy hinchado y cerrado con pus, gime de dolor",
    "lleva horas temblando y no reacciona al llamado",
    "se atragantó, tosió con esfuerzo y quedó decaíd{o}, urgente revisión",
    "fractura evidente en la pata, no apoya nada y chilla",
    "vomitó espuma varias veces y quedó tendid{o}, casi sin respuesta",
    "emergencia: pérdida de conciencia unos segundos durante el paseo",
    "heridas abiertas y sangrantes tras una pelea, está en shock",
    "se niega a moverse, gruñe si lo tocan y respira con mucho esfuerzo",
]

# Colas neutras (idénticas en distribución para las 3 clases, evita fugas)
COLAS = [
    "", "", "", "", ".", " Se continúa monitoreando.", " Anotado por el turno de la tarde.",
    " Sin más comentarios.", " Registro realizado al finalizar la actividad.",
    " Voluntario de turno: Pedro.", " Clima soleado.", " Duración: 30 min.",
]

CONECTORES_MIXTOS = ["pero", "aunque", "sin embargo", "eso sí,", "aun así"]

# ----------------------------------------------------------------------------
# Casos manuales: ambiguos, informales, negaciones, multi-cláusula
# ----------------------------------------------------------------------------
MANUALES = [
    # 0
    ("el Rocky anda feliz hoy, se comió todo y pidió más", 0),
    ("todo bien con el loro, parlanchín como siempre", 0),
    ("la gata no tiene nada raro, solo dormilona", 0),
    ("nada de vómitos, nada de cojera, nada de nada. Un angelito", 0),
    ("se asustó con los petardos pero a los 5 min estaba normal y comió bien", 0),
    ("estuvo tranquilo, tomó agua, comió y no hay novedades que reportar", 0),
    ("no está decaído para nada, al contrario, muy juguetón", 0),
    ("sin signos de dolor, sin sangrado, sin secreciones. Control OK", 0),
    ("jugó x 40 min con los voluntarios, genial!!", 0),
    ("la coneja saltó y exploró todo el corral, buen apetito", 0),
    ("el caballo trotó sin problemas, patas bien y respiración normal", 0),
    ("ayer tenía tos, hoy ya no tose y está como nuevo", 0),
    ("peso igual que la semana pasada, comió toda la verdura", 0),
    ("la tortuga tomó sol y comió lechuga con ganas", 0),
    ("cero problemas hoy, se portó divino", 0),
    # 1
    ("el Rocky está medio raro hoy, no quiso ni el snack", 1),
    ("hoy comió menos pero sigue activo, a ver mañana", 1),
    ("tiene algo de diarrea pero come y toma agua normal", 1),
    ("cojea un poquito, nada grave creo, lo vigilamos", 1),
    ("más dormilón de lo normal, supongo que es el calor", 1),
    ("vomitó 1 vez, después comió y se ve bien. Observar", 1),
    ("no sé, lo noto como apagadito, no tiene fiebre ni nada", 1),
    ("se rasca mucho las orejas, posible otitis leve", 1),
    ("bajó un poco de peso esta semana, apetito irregular", 1),
    ("hoy estuvo irritable con los otros perros, no llegó a morder", 1),
    ("no come mucho pero tampoco está mal, revisar en 2 días", 1),
    ("se cansó rápido y jadeó más de lo normal, luego se recuperó", 1),
    ("estornuda y lagrimea un poco, sin fiebre", 1),
    ("me preocupa un poquito, no comió bien pero juega", 1),
    ("rechazó el medicamento dos veces, al final lo aceptó con comida", 1),
    ("tiene una costra en el lomo, no parece infectada", 1),
    # 2
    ("uff el gato se ve fatal, no se para y no come nada", 2),
    ("no respira bien, abre la boca y se le oye un silbido, urgente!!", 2),
    ("tuvo un ataque, se puso rígido y babeó, llamamos al vet", 2),
    ("la perra no ha tomado agua en 2 días y está seca, muy débil", 2),
    ("tiene un corte profundo y no para de sangrar, lo envolvimos", 2),
    ("vomita todo lo que come desde anoche, ya ni se mueve", 2),
    ("lo encontré tirado, no reaccionaba ni al llamado ni al tacto", 2),
    ("el caballo no apoya la pata y suda mucho, cólico??", 2),
    ("se comió algo del basurero y ahora tiembla y babea, intoxicación", 2),
    ("el loro está en el piso, con las plumas erizadas y no vuela, se ve gravísimo", 2),
    ("abdomen duro e hinchado, llora si lo tocan, no defeca hace días", 2),
    ("tenía buen ánimo en la mañana pero en la tarde colapsó, respira muy rápido", 2),
    ("pese a que come, tiene una hemorragia que no cede y está pálido", 2),
    ("es la tercera vez que convulsiona esta semana", 2),
    ("no quiere comer ni beber, no se levanta y tiene la mirada perdida", 2),
]

# ----------------------------------------------------------------------------
# Casos manuales adicionales (ambiguos). Criterio de etiquetado:
#   2 = signo grave / posible riesgo vital / requiere veterinario urgente
#   1 = molestia leve o cambio que requiere observación
#   0 = sin hallazgos preocupantes (síntomas negados, resueltos o explicados)
# ----------------------------------------------------------------------------
MANUALES_EXTRA = [
    # --- 0: síntomas negados, resueltos o explicados (parecen graves, no lo son)
    ("no hay rastro de la cojera de ayer, corrió como siempre", 0),
    ("estaba muy decaído el lunes pero hoy ya está recuperado y comiendo bien", 0),
    ("ni una gota de sangre, la herida cerró perfecto", 0),
    ("no vomitó, no tuvo diarrea y tampoco tosió. Nada raro", 0),
    ("se hizo el dormido pero cuando vio la comida salió corriendo, apetito de siempre", 0),
    ("descartado el golpe, el vet lo revisó y está sano", 0),
    ("pensé que cojeaba pero era una espina, ya se la saqué y camina normal", 0),
    ("jadea después de correr, normal con este calor, se recuperó enseguida y tomó agua", 0),
    ("duerme mucho pero es un gato viejo y es su rutina, comió bien", 0),
    ("no tiene fiebre, no tiene tos, no tiene nada. Falsa alarma", 0),
    ("la tos de la semana pasada desapareció por completo", 0),
    ("no se observa letargo, responde a todo", 0),
    ("ayer estuvo mal del estómago, hoy heces perfectas y mucho apetito", 0),
    ("el temblor era de emoción al ver a los voluntarios, nada de qué preocuparse", 0),
    ("ya no sangra, la costra está seca y limpia", 0),
    ("ni vómitos ni apatía, solo un poco de pereza después de comer", 0),
    ("respira normal, sin silbidos ni esfuerzo", 0),
    ("mordió un juguete y se asustó, pero no se hizo daño", 0),
    ("tiene hambre de lobo, no se le ve ningún problema", 0),
    ("tuvo un momento de estrés en el baño pero se calmó rápido y comió normal", 0),
    ("sin cambios respecto a ayer, todo estable y bien", 0),
    ("sigue en tratamiento, hoy tomó todas las pastillas y se ve mejor", 0),
    ("xq tan preocupados? se ve perfecto jaja", 0),
    ("no está agresivo, solo ladra de emoción", 0),
    ("ningún síntoma preocupante, el vet dio el alta", 0),
    ("perdió un diente de leche, sin sangrado, come normal", 0),
    ("se ve cansada pero fue el paseo largo, durmió y se recuperó", 0),
    ("convulsiones: ninguna esta semana, sigue con su medicación y estable", 0),
    ("no hay dificultad para respirar ni secreción nasal, todo ok", 0),
    ("el miedo a los truenos ya pasó, volvió a comer y a jugar", 0),
    # --- 1: ambiguos / casos límite leves
    ("no está mal pero tampoco bien, algo le pasa", 1),
    ("come, bebe y juega, pero tiene un ojo un poco rojo", 1),
    ("vomitó dos veces en la mañana, en la tarde estaba mejor y comió", 1),
    ("cojera leve que va y viene, revisar si persiste", 1),
    ("el apetito bajó desde ayer, aun así acepta premios", 1),
    ("tose de vez en cuando, no parece grave", 1),
    ("se le notan las costillas un poco más, vigilar peso", 1),
    ("me da la impresión de que está triste, busca poco contacto", 1),
    ("diarrea suave dos veces, sigue activo y tomando agua", 1),
    ("no hay sangre pero la herida está un poco inflamada", 1),
    ("jadeó más rato de lo normal, pero tomó agua y se calmó", 1),
    ("hoy no quiso salir de su casita, solo salió a comer", 1),
    ("tiene una verruga que parece haber crecido, avisar al vet cuando pueda", 1),
    ("orina con más frecuencia, sin otros signos", 1),
    ("temblaba un poco pero se le pasó, no sé si fue frío", 1),
    ("se mueve más lento al levantarse, probablemente artrosis", 1),
    ("empezó a perder pelo en la cola, sin picazón", 1),
    ("no tomó casi nada de agua en la mañana pero en la tarde sí", 1),
    ("mucha sed hoy, vigilar", 1),
    ("se encorva un poco al caminar, no llora ni nada", 1),
    ("la herida supura un poquito, la limpié con suero", 1),
    ("ya van 2 días con menos apetito, todavía no es alarmante", 1),
    ("llora cuando se queda sola, parece ansiedad por separación", 1),
    ("se rasca sin parar, quizá pulgas, pero de ánimo está bien", 1),
    ("no es grave pero lo noto flojo, mejor observarlo", 1),
    ("estornuda con mocos claros, sin fiebre, controlar", 1),
    ("no come el pienso pero sí la comida húmeda", 1),
    ("hace ruidos raros al respirar pero sin esfuerzo, avisar al vet", 1),
    ("quizá sea nada, pero hoy no respondió igual al llamado", 1),
    ("dolor leve al tocar la cadera, pero camina", 1),
    ("no sangra pero se ve una hinchazón pequeña en la pata, vigilar", 1),
    ("temblor muy leve en una pata al descansar, no sé si es frío", 1),
    # --- 2: graves, algunos con inicio engañoso o lenguaje informal
    ("dice la voluntaria que no se mueve desde la mañana, respira muy lento", 2),
    ("parecía dormida pero no despierta ni con ruido ni con caricias", 2),
    ("aunque comió algo, vomitó sangre después y se ve muy mal", 2),
    ("no hay sangrado visible pero está pálido y frío, parece colapso", 2),
    ("estaba bien en la mañana, a las 3 pm empezó a convulsionar", 2),
    ("se tragó un trozo de plástico y ahora tiene arcadas sin parar, urgente", 2),
    ("la herida que parecía leve ahora está negra, huele fatal y tiene fiebre alta", 2),
    ("no ha orinado en más de 24 horas y llora al intentarlo", 2),
    ("la yegua se tira al suelo y se revuelca, no deja de mirarse el costado", 2),
    ("pasó de decaído a inconsciente en menos de una hora", 2),
    ("mordida profunda en el cuello, mucha sangre, está en shock", 2),
    ("no apoya la pata y no para de gritar", 2),
    ("encías azuladas y respiración con ruido, se está ahogando", 2),
    ("lo atropelló un carro en la entrada, está consciente pero no se levanta", 2),
    ("no es la primera vez que no come, ya van 4 días y está esquelético", 2),
    ("se le cierran los ojos y se desploma, algo muy grave", 2),
    ("abdomen tenso, hace arcadas sin vomitar, posible torsión", 2),
    ("temblores y rigidez en todo el cuerpo, no reacciona", 2),
    ("mucha fiebre, no tolera ni el agua y no se levanta", 2),
    ("aunque no sangra, el golpe fue fuerte y tiene la pupila dilatada", 2),
    ("se cayó del árbol y no mueve las patas traseras", 2),
    ("comió veneno para ratas, vomita y sangra por la nariz", 2),
    ("gime sin parar y se muerde la pata, no deja que nadie se acerque, dolor intenso", 2),
    ("ayer dije que no pasaba nada si comía poco, pero hoy ya ni se levanta ni abre los ojos", 2),
    ("respira con la barriga y tiene la lengua morada", 2),
    ("convulsión de más de 5 minutos, no se recupera", 2),
    ("sangra por la boca y no puede tragar", 2),
    ("no come, no bebe, no juega, la barriga está durísima y llora", 2),
    ("sobredosis de medicamento, babea y está desorientado", 2),
    ("el vet dijo que es urgente, hay que operarlo hoy", 2),
    ("xq no avisaron antes?? lleva dias sin comer y ya no se para", 2),
    ("ta muy mal el pobre, no reacciona y tiene la respiración rarisima", 2),
]

# ----------------------------------------------------------------------------
# Ruido (independiente de la etiqueta)
# ----------------------------------------------------------------------------
ABREV = [(" que ", " q "), (" por ", " x "), (" porque ", " xq "), (" también ", " tmb "),
         (" con ", " c/ "), (" para ", " pa "), (" muy ", " mui "), (" está ", " ta "),
         (" pero ", " pero "), (" más ", " mas "), (" mañana", " mñn"), (" veterinario", " vet")]
ALARGAR = [("muy", "muuuy"), ("bien", "bieeen"), ("mal", "maaal"), ("hoy", "hoyy"), ("nada", "naaada")]
EMOJIS = ["🐶", "🐱", "😟", "🙏", "✅", "⚠️", "😊", "👍", "💔", "🐾", "🥺", "😅"]
PREFIJOS = ["Obs:", "OBS -", "Nota:", "Turno mañana:", "Turno tarde:", "(08:30)", "(15:45)", "#actividad", "Reporte:"]


def quitar_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def typo(s: str, rng: random.Random) -> str:
    palabras = s.split(" ")
    idx = [i for i, w in enumerate(palabras) if len(w) > 4 and w.isalpha()]
    if not idx:
        return s
    i = rng.choice(idx)
    w = list(palabras[i])
    j = rng.randrange(1, len(w) - 2)
    w[j], w[j + 1] = w[j + 1], w[j]
    palabras[i] = "".join(w)
    return " ".join(palabras)


def ensuciar(s: str, rng: random.Random) -> str:
    if rng.random() < 0.15:
        s = quitar_tildes(s)
    if rng.random() < 0.20:
        for a, b in ABREV:
            if rng.random() < 0.7:
                s = s.replace(a, b)
    if rng.random() < 0.10:
        a, b = rng.choice(ALARGAR)
        s = re.sub(rf"\b{a}\b", b, s, count=1)
    if rng.random() < 0.08:
        s = typo(s, rng)
    if rng.random() < 0.10:
        s = s.lower()
    if rng.random() < 0.05:
        s = s.upper()
    if rng.random() < 0.10:
        s = s.rstrip(".") + rng.choice(["!!", "...", "!!!", " ??", ".."])
    if rng.random() < 0.12:
        s = s + " " + rng.choice(EMOJIS)
    if rng.random() < 0.06:
        s = rng.choice(PREFIJOS) + " " + s
    if rng.random() < 0.10:
        s = "  " + s.replace(" ", "  ", 1) + " "
    if rng.random() < 0.03:
        s = s.replace(". ", ".\n", 1)
    return s


# ----------------------------------------------------------------------------
# Generación
# ----------------------------------------------------------------------------
def capitalizar(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def construir(rng: random.Random, frases_por_clase: dict, etiqueta: int) -> str:
    n, esp, o = rng.choice(ANIMALES)
    apertura = rng.choice(APERTURAS).format(
        n=n, esp=esp.split(" ", 1)[1], Esp=capitalizar(esp))
    f1 = rng.choice(frases_por_clase[etiqueta])
    cuerpo = f1
    # Frases mixtas: la gravedad total es la máxima de las cláusulas
    if rng.random() < 0.18 and etiqueta in (1, 2):
        menor = rng.choice(frases_por_clase[etiqueta - 1])
        cuerpo = f"{menor} {rng.choice(CONECTORES_MIXTOS)} {f1}"
    texto = f"{apertura} {cuerpo}".replace("{o}", o)
    return texto.strip() + rng.choice(COLAS)


def main(n_plantilla=(260, 165, 110)) -> None:
    rng = random.Random(SEED)
    # Sustituye {o} dentro de las frases en construir(); preparar copias
    frases = {0: CLASE_0, 1: CLASE_1, 2: CLASE_2}
    filas, vistos = [], set()

    def agregar(txt, y):
        clave = re.sub(r"\W+", "", quitar_tildes(txt.lower()))
        if clave in vistos:
            return False
        vistos.add(clave)
        filas.append((txt, y))
        return True

    for txt, y in MANUALES + MANUALES_EXTRA:
        agregar(ensuciar(txt, rng) if rng.random() < 0.5 else txt, y)

    for y, n in enumerate(n_plantilla):
        creados, intentos = 0, 0
        while creados < n and intentos < n * 30:
            intentos += 1
            t = construir(rng, frases, y)
            if agregar(ensuciar(t, rng), y):
                creados += 1

    rng.shuffle(filas)
    df = pd.DataFrame(filas, columns=["texto", "etiqueta"])

    # Suciedad deliberada de la tabla: duplicados exactos, textos vacíos y muy cortos
    dup = df.sample(8, random_state=SEED)
    vacios = pd.DataFrame({"texto": ["", "   ", "\n", None], "etiqueta": [0, 1, 0, 2]})
    cortos = pd.DataFrame({"texto": ["ok", "bien", "mal", "."], "etiqueta": [0, 0, 1, 1]})
    df = pd.concat([df, dup, vacios, cortos], ignore_index=True).sample(frac=1, random_state=SEED)
    df.insert(0, "id", range(1, len(df) + 1))

    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_RAW, index=False, encoding="utf-8")
    print(f"Guardado {DATA_RAW} con {len(df)} filas")
    print(df["etiqueta"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
