{{- define "hf-proxy-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "hf-proxy-app.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "hf-proxy-app.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "hf-proxy-app.labels" -}}
helm.sh/chart: {{ include "hf-proxy-app.chart" . }}
{{ include "hf-proxy-app.selectorLabels" . }}
{{- if .Values.apolo_app_id }}
platform.apolo.us/app-id: {{ .Values.apolo_app_id | quote }}
{{- end }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "hf-proxy-app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hf-proxy-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}

{{- end }}

{{- define "hf-proxy-app.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "hf-proxy-app.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
