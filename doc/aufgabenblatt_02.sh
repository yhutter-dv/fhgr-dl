#!/bin/bash
pandoc --template ./eisvogel.tex aufgabenblatt_02.md --number-sections --listings --pdf-engine=lualatex  -o ./pdfs/Aufgabenblatt_02.pdf 
