# TP1

## Requerimientos <a name="requerimientos"></a>

Debe instalar

- [Python >= v3.10](https://www.python.org/downloads/)
- [Poetry >= v1.0.0](https://python-poetry.org/)

## Instalación <a name="instalacion"></a>

```bash
poetry install
```

## Ejecución <a name="ejecución"></a>

Puede ejecutar el ladoA usando poetry mediante:

```bash
poetry run ladoA
```

Aquí, si desea, puede pasarle ciertos parámetros. Para imprimir la ayuda puede correr:

```bash
poetry run ladoA -h
```

Por otro lado, para el ladoB:

```bash
poetry run ladoB
```

Note que para el ladoB se puede modificar el `config.json` de la carpeta ladoB para cambiar los parametros.

## Testeos <a name="tests"></a>

Debe entrar a la carpeta `tests/ladoA` y correr:

```bash
poetry run pytest test_ladoA.py
```

Por otro lado, para correr los testeos del ladoB debe entrar a `tests/ladoB` y correr

```bash
poetry run pytest test_everything.py
```

## Autores
- Barmasch, Juan Martín (61033)
- Bellver, Ezequiel (61268)
- Castagnino, Salvador (60590)
- Lo Coco, Santiago (61301)
- Negro, Juan Manuel (61225)
