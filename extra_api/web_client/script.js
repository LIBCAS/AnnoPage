const CLIENT_CONFIG = {
    apiUrl: "/api/v1",
    apiKeyQueryParam: "api_key",
    apiKeyHeader: "X-API-Key",
    pollingIntervalMs: 1500,
    endpoints: {
        createJob: "/jobs",
        jobStatus: "/jobs/{jobId}",
        uploadImage: "/jobs/{jobId}/images/{imageName}/files/image",
        uploadAltoXml: "/jobs/{jobId}/images/{imageName}/files/alto",
        uploadPageXml: "/jobs/{jobId}/images/{imageName}/files/page",
        resultZip: "/jobs/{jobId}/result"
    },
    uploadFieldNames: {
        file: "file"
    },
    defaultCaptioningProvider: "OpenRouter",
    defaultCaptioningModel: "openai/gpt-4.1-mini",
    captioningProviders: [
        {
            label: "OpenAI",
            value: "OpenAI"
        },
        {
            label: "OpenRouter",
            value: "OpenRouter"
        },
        {
            label: "CERIT",
            value: "CERIT"
        }
    ]
};

const form = document.querySelector("#job-form");
const providerSelect = document.querySelector("#captioning-provider");
const modelInput = document.querySelector("#captioning-model");
const captioningApiKeyInput = document.querySelector("#captioning-api-key");
const imageFileInput = document.querySelector("#image-file");
const xmlFileInput = document.querySelector("#xml-file");
const xmlTypeInputs = Array.from(document.querySelectorAll("input[name='xmlType']"));
const submitButton = document.querySelector("#submit-button");
const statusLabel = document.querySelector("#job-status");
const zipDownload = document.querySelector("#zip-download");
const resultView = document.querySelector("#result-view");
const originalImage = document.querySelector("#original-image");
const originalXmlOutput = document.querySelector("#original-xml-output");
const renderImage = document.querySelector("#render-image");
const renderDownload = document.querySelector("#render-download");
const xmlOutput = document.querySelector("#xml-output");
const xmlDownload = document.querySelector("#xml-download");
const logOutput = document.querySelector("#log-output");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

let originalImageObjectUrl = null;
let zipObjectUrl = null;
let renderObjectUrl = null;
let xmlObjectUrl = null;
const initialApiKey = new URLSearchParams(window.location.search).get(CLIENT_CONFIG.apiKeyQueryParam)?.trim() ?? "";

function setStatus(text) {
    statusLabel.textContent = text;
}

function logMessage(text, kind = "") {
    logOutput.textContent = text;
    logOutput.className = `log-output ${kind}`.trim();
}

function appendMessage(text) {
    logOutput.textContent = `${logOutput.textContent}${logOutput.textContent ? "\n" : ""}${text}`;
}

function normalizeBaseUrl(url) {
    return url.replace(/\/+$/, "");
}

function endpointUrl(name, values = {}) {
    const template = CLIENT_CONFIG.endpoints[name];
    if (!template) {
        throw new Error(`Missing endpoint config for '${name}'.`);
    }

    const path = template.replace(/\{(\w+)\}/g, (_, key) => encodeURIComponent(values[key] ?? ""));
    return `${normalizeBaseUrl(CLIENT_CONFIG.apiUrl)}${path}`;
}

function apiHeaders(extraHeaders = {}) {
    const headers = { ...extraHeaders };
    if (initialApiKey) {
        headers[CLIENT_CONFIG.apiKeyHeader] = initialApiKey;
    }
    return headers;
}

async function apiFetch(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: apiHeaders(options.headers)
    });

    if (!response.ok) {
        const body = await response.text();
        throw new Error(`${response.status} ${response.statusText}${body ? `: ${body}` : ""}`);
    }

    return response;
}

function getResponseData(payload) {
    if (payload && typeof payload === "object" && "data" in payload) {
        return payload.data;
    }
    return payload;
}

function getJobId(payload) {
    const data = getResponseData(payload);
    return data?.id ?? data?.job_id ?? payload?.id ?? payload?.job_id;
}

function getJobStatus(payload) {
    const data = getResponseData(payload);
    return String(data?.status ?? data?.state ?? payload?.status ?? payload?.state ?? "").toLowerCase();
}

function isFinishedStatus(status) {
    return ["done", "finished", "completed", "complete", "success", "succeeded"].includes(status);
}

function isFailedStatus(status) {
    return ["failed", "failure", "error", "errored", "cancelled", "canceled"].includes(status);
}

function populateSelect(select, items, valueKey = "id", labelKey = "label") {
    select.innerHTML = "";
    for (const item of items) {
        const option = document.createElement("option");
        option.value = item[valueKey] ?? "";
        option.textContent = item[labelKey] ?? item[valueKey] ?? "";
        select.appendChild(option);
    }
}

function selectedProvider() {
    return CLIENT_CONFIG.captioningProviders.find((provider) => provider.value === providerSelect.value);
}

function selectedXmlType() {
    return xmlTypeInputs.find((input) => input.checked)?.value ?? "alto";
}

function updateXmlTypeState() {
    const hasXml = xmlFileInput.files.length > 0;
    for (const input of xmlTypeInputs) {
        input.disabled = !hasXml;
    }
}

function activateTab(tabName) {
    for (const button of tabButtons) {
        const isActive = button.dataset.tab === tabName;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-selected", String(isActive));
    }

    for (const panel of tabPanels) {
        panel.classList.toggle("active", panel.dataset.panel === tabName);
    }
}

function setBusy(isBusy) {
    submitButton.disabled = isBusy;
}

function setZipDownload(blob = null, jobId = null) {
    if (zipObjectUrl) {
        URL.revokeObjectURL(zipObjectUrl);
        zipObjectUrl = null;
    }

    if (blob) {
        zipObjectUrl = URL.createObjectURL(blob);
        zipDownload.href = zipObjectUrl;
        zipDownload.download = jobId ? `${jobId}.zip` : "annopage-results.zip";
        zipDownload.classList.remove("disabled");
        zipDownload.setAttribute("aria-disabled", "false");
    } else {
        zipDownload.removeAttribute("href");
        zipDownload.download = "annopage-results.zip";
        zipDownload.classList.add("disabled");
        zipDownload.setAttribute("aria-disabled", "true");
    }
}

function resetResultView() {
    resultView.classList.add("hidden");
    originalImage.removeAttribute("src");
    originalXmlOutput.value = "";
    renderImage.removeAttribute("src");
    xmlOutput.value = "";
    logOutput.textContent = "";
    logOutput.className = "log-output";
    setZipDownload();

    if (originalImageObjectUrl) {
        URL.revokeObjectURL(originalImageObjectUrl);
        originalImageObjectUrl = null;
    }

    if (renderObjectUrl) {
        URL.revokeObjectURL(renderObjectUrl);
        renderObjectUrl = null;
    }

    if (xmlObjectUrl) {
        URL.revokeObjectURL(xmlObjectUrl);
        xmlObjectUrl = null;
    }
}

function buildEngineSettings() {
    const settings = {
        outputs: {
            alto: true,
            embeddings: true,
            embeddings_jsonlines: true,
            renders: true,
            crops: true,
            image_captioning_prompts: true
        }
    };

    const provider = selectedProvider();
    const captioningApiKey = captioningApiKeyInput.value.trim();
    const captioningModel = modelInput.value.trim();

    if (provider || captioningApiKey || captioningModel) {
        settings.image_captioning = {
            api: provider?.value,
            model: captioningModel
        };

        if (captioningApiKey) {
            settings.image_captioning.api_key = captioningApiKey;
        }
    }

    return settings;
}

function buildJobPayload() {
    const imageFile = imageFileInput.files[0];
    const hasXml = xmlFileInput.files.length > 0;
    const xmlType = selectedXmlType();
    const payload = {
        images: [
            {
                name: imageFile.name,
                order: 0
            }
        ],
        engine_settings: buildEngineSettings(),
    };

    if (hasXml && xmlType === "alto") {
        payload.alto_required = true;
    }

    if (hasXml && xmlType === "page") {
        payload.page_required = true;
    }

    return payload;
}

async function createJob() {
    const response = await apiFetch(endpointUrl("createJob"), {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(buildJobPayload())
    });
    const payload = await response.json();
    const jobId = getJobId(payload);

    if (!jobId) {
        throw new Error("The create-job response did not contain a job id.");
    }

    return jobId;
}

async function uploadImageFile(endpointName, jobId, imageName, file) {
    const formData = new FormData();
    formData.append(CLIENT_CONFIG.uploadFieldNames.file, file, file.name);

    await apiFetch(endpointUrl(endpointName, { jobId, imageName }), {
        method: "PUT",
        body: formData
    });
}

async function uploadInputs(jobId) {
    const imageFile = imageFileInput.files[0];
    const xmlFile = xmlFileInput.files[0];

    await uploadImageFile("uploadImage", jobId, imageFile.name, imageFile);
    appendMessage(`Uploaded image: ${imageFile.name}`);

    if (xmlFile) {
        const endpointName = selectedXmlType() === "page" ? "uploadPageXml" : "uploadAltoXml";
        await uploadImageFile(endpointName, jobId, imageFile.name, xmlFile);
        appendMessage(`Uploaded XML: ${xmlFile.name}`);
    }
}

async function pollJob(jobId) {
    while (true) {
        const response = await apiFetch(endpointUrl("jobStatus", { jobId }));
        const payload = await response.json();
        const status = getJobStatus(payload) || "unknown";

        setStatus(status);
        appendMessage(`Job status: ${status}`);

        if (isFinishedStatus(status)) {
            return payload;
        }

        if (isFailedStatus(status)) {
            throw new Error(`Job failed with status '${status}'.`);
        }

        await new Promise((resolve) => setTimeout(resolve, CLIENT_CONFIG.pollingIntervalMs));
    }
}

async function downloadResultZip(jobId) {
    const response = await apiFetch(endpointUrl("resultZip", { jobId }));
    return response.blob();
}

async function showOriginalInputs() {
    const imageFile = imageFileInput.files[0];
    const xmlFile = xmlFileInput.files[0];

    if (imageFile) {
        originalImageObjectUrl = URL.createObjectURL(imageFile);
        originalImage.src = originalImageObjectUrl;
    }

    originalXmlOutput.value = xmlFile ? await xmlFile.text() : "No original XML file selected.";
    resultView.classList.remove("hidden");
    activateTab("original-image");
}

function findFirstZipFile(zip, predicate) {
    return Object.values(zip.files).find((file) => !file.dir && predicate(file.name));
}

async function displayResults(zipBlob) {
    if (!window.JSZip) {
        throw new Error("JSZip is not loaded. Check the script tag in index.html.");
    }

    const zip = await JSZip.loadAsync(zipBlob);
    const renderEntry = findFirstZipFile(zip, (name) => /(^|\/)renders\/.+\.(png|jpe?g|webp)$/i.test(name));
    const xmlEntry = findFirstZipFile(zip, (name) => /(^|\/)(alto|xml|page_xmls?)\/.+\.xml$/i.test(name))
        ?? findFirstZipFile(zip, (name) => /\.xml$/i.test(name));

    if (!renderEntry) {
        throw new Error("The result ZIP does not contain a render image in a renders/ directory.");
    }

    if (!xmlEntry) {
        throw new Error("The result ZIP does not contain an XML file.");
    }

    const renderBlob = await renderEntry.async("blob");
    const xmlText = await xmlEntry.async("text");
    const xmlBlob = new Blob([xmlText], { type: "application/xml" });

    renderObjectUrl = URL.createObjectURL(renderBlob);
    xmlObjectUrl = URL.createObjectURL(xmlBlob);

    renderImage.src = renderObjectUrl;
    renderDownload.href = renderObjectUrl;
    renderDownload.download = renderEntry.name.split("/").pop() || "render.jpg";

    xmlOutput.value = xmlText;
    xmlDownload.href = xmlObjectUrl;
    xmlDownload.download = xmlEntry.name.split("/").pop() || "result.xml";

    resultView.classList.remove("hidden");
    activateTab("render");
}

async function runJob(event) {
    event.preventDefault();
    resetResultView();
    setBusy(true);
    setStatus("Starting");
    logMessage("Creating job...");

    try {
        await showOriginalInputs();
        const jobId = await createJob();
        appendMessage(`Created job: ${jobId}`);

        setStatus("Uploading");
        await uploadInputs(jobId);

        setStatus("Waiting");
        appendMessage("Waiting for processing to finish...");
        await pollJob(jobId);

        setStatus("Downloading");
        appendMessage("Downloading result ZIP...");
        const zipBlob = await downloadResultZip(jobId);
        setZipDownload(zipBlob, jobId);

        setStatus("Extracting");
        appendMessage("Extracting render and XML...");
        await displayResults(zipBlob);

        setStatus("Done");
    } catch (error) {
        setStatus("Error");
        logMessage(error.message, "error");
    } finally {
        setBusy(false);
    }
}

function init() {
    populateSelect(providerSelect, CLIENT_CONFIG.captioningProviders, "value", "label");
    providerSelect.value = CLIENT_CONFIG.defaultCaptioningProvider;
    modelInput.value = CLIENT_CONFIG.defaultCaptioningModel;
    updateXmlTypeState();

    xmlFileInput.addEventListener("change", updateXmlTypeState);
    for (const button of tabButtons) {
        button.addEventListener("click", () => activateTab(button.dataset.tab));
    }
    form.addEventListener("submit", runJob);

    if (!initialApiKey) {
        submitButton.disabled = true;
        setStatus("Missing API key");
        logMessage(`Open this page with ?${CLIENT_CONFIG.apiKeyQueryParam}=YOUR_API_KEY in the URL to enable job submission.`, "error");
    }
}

init();
