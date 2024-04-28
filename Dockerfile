FROM ubuntu:22.04

COPY . .

CMD python /Server/webcrawler.py