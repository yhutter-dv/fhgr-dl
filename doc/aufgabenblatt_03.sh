#!/bin/bash
pandoc --template ./eisvogel.tex aufgabenblatt_03.md --number-sections --listings --pdf-engine=lualatex  -o ./pdfs/Aufgabenblatt_03.pdf 
