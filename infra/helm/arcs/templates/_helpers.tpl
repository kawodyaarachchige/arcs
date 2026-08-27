{{- define "arcs.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "arcs.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "arcs.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: arcs-testbed
{{- end }}

{{- define "arcs.echoLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-echo-service
{{- end }}

{{- define "arcs.cpuLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-cpu-service
{{- end }}

{{- define "arcs.aggregatorLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-aggregator-service
{{- end }}

{{- define "arcs.toxiproxyLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-toxiproxy
{{- end }}

{{- define "arcs.policyLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-policy-service
{{- end }}

{{- define "arcs.envoyLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-envoy
{{- end }}

{{- define "arcs.telemetryBridgeLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-telemetry-bridge
{{- end }}

{{- define "arcs.kafkaLabels" -}}
{{ include "arcs.labels" . }}
app.kubernetes.io/name: arcs-kafka
{{- end }}

{{- define "arcs.aggregatorDownstreamUrls" -}}
{{- if .Values.aggregator.downstreamUrls -}}
{{- .Values.aggregator.downstreamUrls -}}
{{- else -}}
{{- printf "http://%s-toxiproxy:8666/health/ready,http://%s-toxiproxy:8667/health/ready" (include "arcs.fullname" .) (include "arcs.fullname" .) -}}
{{- end -}}
{{- end }}
