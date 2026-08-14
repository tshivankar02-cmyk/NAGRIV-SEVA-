/* =====================================================
   Worker Panel JavaScript — Nagrik-Seva / AI Smart City
   ===================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // 1. File Upload Preview & Drag-and-Drop
  const dropZone = document.getElementById('imageDropZone');
  const fileInput = document.getElementById('after_image');
  const previewContainer = document.getElementById('uploadPreviewContainer');
  const previewImg = document.getElementById('previewImg');
  const removeImgBtn = document.getElementById('removeImgBtn');
  const base64Input = document.getElementById('after_image_base64');

  if (dropZone && fileInput) {
    dropZone.addEventListener('click', (e) => {
      if (e.target !== removeImgBtn && !e.target.closest('#removeImgBtn')) {
        fileInput.click();
      }
    });

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
      alert('Please select an image file (JPG, PNG, WEBP).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File size exceeds 10MB limit.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) {
        previewImg.src = e.target.result;
        previewContainer.classList.remove('d-none');
        if (base64Input) base64Input.value = ''; // clear camera base64 if file picked
      }
    };
    reader.readAsDataURL(file);
  }

  if (removeImgBtn) {
    removeImgBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (fileInput) fileInput.value = '';
      if (base64Input) base64Input.value = '';
      if (previewContainer) previewContainer.classList.add('d-none');
    });
  }

  // 2. Camera Capture Support
  const startCameraBtn = document.getElementById('startCameraBtn');
  const cameraModal = document.getElementById('cameraModal');
  const cameraVideo = document.getElementById('cameraVideo');
  const capturePhotoBtn = document.getElementById('capturePhotoBtn');
  let mediaStream = null;

  if (startCameraBtn && cameraVideo) {
    startCameraBtn.addEventListener('click', async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' }
        });
        cameraVideo.srcObject = mediaStream;
        const modal = new bootstrap.Modal(cameraModal);
        modal.show();
      } catch (err) {
        alert('Could not access camera: ' + err.message);
      }
    });

    if (cameraModal) {
      cameraModal.addEventListener('hidden.bs.modal', () => {
        if (mediaStream) {
          mediaStream.getTracks().forEach(track => track.stop());
          mediaStream = null;
        }
      });
    }

    if (capturePhotoBtn) {
      capturePhotoBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = cameraVideo.videoWidth || 640;
        canvas.height = cameraVideo.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');

        if (base64Input) base64Input.value = dataUrl;
        if (fileInput) fileInput.value = '';
        if (previewImg) {
          previewImg.src = dataUrl;
          previewContainer.classList.remove('d-none');
        }

        const modal = bootstrap.Modal.getInstance(cameraModal);
        if (modal) modal.hide();
      });
    }
  }

  // 3. Custom Tool & Problem Tag Synchronizer
  const customToolInput = document.getElementById('customToolInput');
  const addCustomToolBtn = document.getElementById('addCustomToolBtn');
  const toolsGroup = document.getElementById('toolsSelectorGroup');

  if (addCustomToolBtn && customToolInput && toolsGroup) {
    addCustomToolBtn.addEventListener('click', () => {
      const toolName = customToolInput.value.trim();
      if (!toolName) return;

      const safeId = 'tool_custom_' + Date.now();
      const div = document.createElement('div');
      div.className = 'd-inline-block me-2 mb-2';
      div.innerHTML = `
        <input type="checkbox" class="tag-checkbox tool-checkbox" id="${safeId}" value="${toolName}" checked>
        <label for="${safeId}" class="tag-label"><i class="fas fa-wrench me-1"></i>${toolName}</label>
      `;
      toolsGroup.appendChild(div);
      customToolInput.value = '';
      updateToolsField();
    });
  }

  // Sync checkboxes to hidden input text fields
  const repairForm = document.getElementById('repairReportForm');
  if (repairForm) {
    repairForm.addEventListener('submit', (e) => {
      updateToolsField();
      updateProblemsField();
      updateTeamField();

      const base64Val = base64Input ? base64Input.value : '';
      const fileVal = fileInput && fileInput.files.length > 0;

      if (!base64Val && !fileVal) {
        e.preventDefault();
        alert('Please upload or capture an After-Repair Image before submitting.');
      }
    });
  }

  function updateToolsField() {
    const checked = Array.from(document.querySelectorAll('.tool-checkbox:checked')).map(cb => cb.value);
    const hiddenTools = document.getElementById('tools_used_hidden');
    if (hiddenTools) hiddenTools.value = checked.join(', ');
  }

  function updateProblemsField() {
    const checked = Array.from(document.querySelectorAll('.problem-checkbox:checked')).map(cb => cb.value);
    const customText = document.getElementById('problems_faced_text') ? document.getElementById('problems_faced_text').value.trim() : '';
    const hiddenProblems = document.getElementById('problems_faced_hidden');
    if (hiddenProblems) {
      let combined = checked.join('; ');
      if (customText) {
        combined += (combined ? ' — ' : '') + customText;
      }
      hiddenProblems.value = combined;
    }
  }

  function updateTeamField() {
    const checked = Array.from(document.querySelectorAll('.team-checkbox:checked')).map(cb => cb.value);
    const customText = document.getElementById('team_custom_text') ? document.getElementById('team_custom_text').value.trim() : '';
    const hiddenTeam = document.getElementById('team_members_hidden');
    if (hiddenTeam) {
      let combined = checked.join(', ');
      if (customText) {
        combined += (combined ? ', ' : '') + customText;
      }
      hiddenTeam.value = combined;
    }
  }
});
