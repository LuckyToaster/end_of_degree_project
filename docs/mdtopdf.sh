#!/bin/bash
pandoc "$1" -o "${1%.*}.pdf" --template eisvogel --listings --pdf-engine=lualatex --toc --number-sections --resource-path=.:./images
