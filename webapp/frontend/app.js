/**
 * VisaChronos - AI Visa Processing Time Predictor
 * Frontend JavaScript
 * Author: Rahul Makwana
 * Infosys Springboard Project
 */

// API Configuration
const API_BASE = '';
let lastFormData = null; // Store last form submission for What-If / Optimal Month

// ============================================
// THEME TOGGLE
// ============================================
function initTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;

    // Check saved preference or system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }

    themeToggle.addEventListener('click', function () {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        // Add smooth transition
        document.body.style.transition = 'background 0.3s ease, color 0.3s ease';
    });
}

// ============================================
// SCROLL REVEAL ANIMATIONS
// ============================================
function initScrollReveal() {
    const revealElements = document.querySelectorAll('.reveal');

    if (!revealElements.length) return;

    const revealOnScroll = () => {
        revealElements.forEach(el => {
            const elementTop = el.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;

            if (elementTop < windowHeight - 100) {
                el.classList.add('visible');
            }
        });
    };

    // Initial check
    revealOnScroll();

    // On scroll
    window.addEventListener('scroll', revealOnScroll, { passive: true });
}

// ============================================
// FORM HANDLING
// ============================================
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const loadingEl = document.getElementById('loading');
const resultsPanel = document.getElementById('results-panel');

if (form) {
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Show loading, hide results
        showLoading(true);
        hideResults();

        // Collect form data
        const formData = {
            applicant_age: parseInt(document.getElementById('age').value),
            nationality: document.getElementById('nationality').value,
            visa_type: document.getElementById('visa_type').value,
            occupation: document.getElementById('occupation').value,
            education_level: document.getElementById('education').value,
            duration_requested_days: parseInt(document.getElementById('duration').value) || 30,
            num_previous_visits: parseInt(document.getElementById('previous_visits').value) || 0,
            financial_proof_usd: parseFloat(document.getElementById('financial_proof').value) || 15000,
            has_sponsor: document.getElementById('has_sponsor').checked,
            documents_complete: document.getElementById('documents_complete').checked,
            express_processing: document.getElementById('express_processing').checked,
            application_month: parseInt(document.getElementById('application_month').value) || 1
        };
        lastFormData = { ...formData };

        try {
            const response = await fetch(API_BASE + '/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            showLoading(false);
            displayResults(result);

        } catch (error) {
            console.error('Error:', error);
            showLoading(false);
            alert('An error occurred. Please try again.');
        }
    });
}

// ============================================
// DISPLAY RESULTS
// ============================================
function displayResults(data) {
    // Main prediction
    document.getElementById('result-days').innerHTML =
        data.predicted_days + ' <span>days</span>';

    document.getElementById('result-range').textContent =
        'Estimated range: ' + data.min_days + ' to ' + data.max_days + ' days';

    // Risk level
    const riskEl = document.getElementById('result-risk');
    riskEl.textContent = data.risk_level;
    riskEl.className = 'result-item__value ' + getRiskClass(data.risk_level);

    // Approval likelihood
    const approvalEl = document.getElementById('result-approval');
    approvalEl.textContent = data.approval_likelihood + ' (' + data.approval_percentage + '%)';
    approvalEl.className = 'result-item__value ' + getApprovalClass(data.approval_likelihood);

    // Averages
    document.getElementById('result-country-avg').textContent =
        data.country_average + ' days';
    document.getElementById('result-visa-avg').textContent =
        data.visa_type_average + ' days';

    // Risk gauge
    updateRiskGauge(data.risk_score);

    // Key factors
    displayFactors(data.factors);

    // Timeline visualization
    displayTimeline(data.predicted_days, data.min_days, data.max_days);

    // NEW: Animate model confidence gauge
    animateConfidenceGauge();

    // NEW: Update comparison chart with actual values
    updateComparisonChart(data.predicted_days, data.visa_type_average, data.country_average);

    // Save to history
    saveToHistory(data);

    // Show results
    showResults();

    // NEW: AI-powered features
    generateAISummary(data, lastFormData);
    renderWaterfallChart(data);
    runWhatIfAnalysis(lastFormData, data);
    findOptimalMonth(lastFormData);
}

// Update the risk gauge bar
function updateRiskGauge(riskScore) {
    const gauge = document.getElementById('risk-gauge-fill');
    if (!gauge) return;

    // Risk score is 0-5+, map to percentage
    const percentage = Math.min((riskScore / 5) * 100, 100);
    gauge.style.width = percentage + '%';

    // Set color based on risk
    gauge.classList.remove('low', 'medium', 'high');
    if (riskScore <= 1) {
        gauge.classList.add('low');
    } else if (riskScore <= 3) {
        gauge.classList.add('medium');
    } else {
        gauge.classList.add('high');
    }
}

// Display key factors
function displayFactors(factors) {
    const container = document.getElementById('result-factors');
    if (!container) return;

    container.innerHTML = '';

    if (!factors) return;

    const factorLabels = {
        peak_season: 'Peak Season',
        documents_complete: 'Documents Complete',
        has_sponsor: 'Has Sponsor',
        express_processing: 'Express Processing',
        previous_visits: 'Previous Visits',
        financial_proof: 'Financial Proof'
    };

    for (const [key, value] of Object.entries(factors)) {
        const label = factorLabels[key] || key.replace(/_/g, ' ');
        const displayValue = typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value;

        const factorEl = document.createElement('div');
        factorEl.className = 'factor';
        factorEl.innerHTML =
            '<span class="factor__label">' + label + '</span>' +
            '<span class="factor__value">' + displayValue + '</span>';
        container.appendChild(factorEl);
    }
}

// Display timeline visualization
function displayTimeline(predictedDays, minDays, maxDays) {
    const timelineContainer = document.getElementById('timeline-container');
    if (!timelineContainer) return;

    const today = new Date();
    const expectedDate = new Date(today);
    expectedDate.setDate(expectedDate.getDate() + Math.round(predictedDays));

    const maxDate = new Date(today);
    maxDate.setDate(maxDate.getDate() + Math.round(maxDays));

    const formatDate = (date) => {
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        });
    };

    timelineContainer.innerHTML = `
        <div class="timeline">
            <div class="timeline__track">
                <div class="timeline__progress" style="width: 50%"></div>
            </div>
            <div class="timeline__points">
                <div class="timeline__point timeline__point--start">
                    <div class="timeline__marker"></div>
                    <div class="timeline__label">Application</div>
                    <div class="timeline__date">${formatDate(today)}</div>
                </div>
                <div class="timeline__point timeline__point--mid">
                    <div class="timeline__marker timeline__marker--active"></div>
                    <div class="timeline__label">Expected</div>
                    <div class="timeline__date">${formatDate(expectedDate)}</div>
                </div>
                <div class="timeline__point timeline__point--end">
                    <div class="timeline__marker"></div>
                    <div class="timeline__label">Latest</div>
                    <div class="timeline__date">${formatDate(maxDate)}</div>
                </div>
            </div>
        </div>
        <div class="timeline__summary">
            <p>If you apply today, expect your visa by <strong>${formatDate(expectedDate)}</strong></p>
        </div>
    `;
}

// ============================================
// HELPER FUNCTIONS
// ============================================
function getRiskClass(riskLevel) {
    if (riskLevel === 'Low') return 'success';
    if (riskLevel === 'Medium') return 'warning';
    return 'danger';
}

function getApprovalClass(likelihood) {
    if (likelihood === 'High') return 'success';
    if (likelihood === 'Medium') return 'warning';
    return 'danger';
}

function showLoading(show) {
    if (loadingEl) {
        loadingEl.classList.toggle('visible', show);
    }
    if (submitBtn) {
        submitBtn.disabled = show;
        submitBtn.textContent = show ? 'Processing...' : 'Get Processing Estimate';
    }
}

function showResults() {
    if (resultsPanel) {
        resultsPanel.classList.add('visible');
        // Smooth scroll to results
        setTimeout(function () {
            resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
}

function hideResults() {
    if (resultsPanel) {
        resultsPanel.classList.remove('visible');
    }
}

// Reset form
function resetForm() {
    if (form) {
        form.reset();
    }
    hideResults();
}

// Reset and scroll to top
function resetAndScroll() {
    resetForm();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Set current month as default
function setDefaultMonth() {
    const monthSelect = document.getElementById('application_month');
    if (monthSelect) {
        const currentMonth = new Date().getMonth() + 1;
        monthSelect.value = currentMonth;
    }
}

// ============================================
// AI VISUALIZATION ANIMATIONS
// ============================================

// Animate the model confidence circular gauge
function animateConfidenceGauge() {
    const circle = document.getElementById('confidence-circle');
    if (!circle) return;

    // 77% accuracy - animate to show this
    const accuracy = 77;
    const circumference = 339.3; // 2 * PI * 54
    const offset = circumference - (accuracy / 100) * circumference;

    // Delay animation slightly for visual effect
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
    }, 300);
}

// Update the comparison chart with actual prediction values
function updateComparisonChart(yourEstimate, visaAvg, countryAvg) {
    const overallAvg = 8.2; // Historical overall average
    const maxValue = Math.max(yourEstimate, visaAvg, countryAvg, overallAvg, 15);

    // Helper to set bar and value
    const setBar = (barId, valueId, value) => {
        const bar = document.getElementById(barId);
        const valueEl = document.getElementById(valueId);
        if (bar && valueEl) {
            const percentage = (value / maxValue) * 100;
            setTimeout(() => {
                bar.style.width = percentage + '%';
            }, 500);
            valueEl.textContent = value.toFixed(1) + ' days';
        }
    };

    setBar('your-estimate-bar', 'your-estimate-value', yourEstimate);
    setBar('visa-avg-bar', 'visa-avg-value', visaAvg);
    setBar('country-avg-bar', 'country-avg-value', countryAvg);
    setBar('overall-avg-bar', 'overall-avg-value', overallAvg);
}

// ============================================
// PDF EXPORT
// ============================================
function downloadPDF() {
    // Use browser's print functionality to save as PDF
    const originalTitle = document.title;
    document.title = 'VisaChronos_Prediction_' + new Date().toISOString().split('T')[0];

    // Hide elements that shouldn't be in PDF
    const elementsToHide = document.querySelectorAll('.pdf-btn, .btn--outline, .history-panel, .trends-section');
    elementsToHide.forEach(el => el.style.display = 'none');

    window.print();

    // Restore elements
    elementsToHide.forEach(el => el.style.display = '');
    document.title = originalTitle;
}

// ============================================
// PREDICTION HISTORY
// ============================================
const HISTORY_KEY = 'visachronos_history';
const MAX_HISTORY = 10;

// Save prediction to history
function saveToHistory(data) {
    const history = getHistory();
    const entry = {
        id: Date.now(),
        date: new Date().toLocaleString(),
        visa_type: document.getElementById('visa_type')?.value || 'Unknown',
        nationality: document.getElementById('nationality')?.value || 'Unknown',
        predicted_days: data.predicted_days,
        risk_level: data.risk_level,
        approval_percentage: data.approval_percentage
    };

    history.unshift(entry);

    // Keep only last MAX_HISTORY entries
    if (history.length > MAX_HISTORY) {
        history.pop();
    }

    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory();
}

// Get history from localStorage
function getHistory() {
    try {
        const data = localStorage.getItem(HISTORY_KEY);
        return data ? JSON.parse(data) : [];
    } catch (e) {
        return [];
    }
}

// Render history in the UI
function renderHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;

    const history = getHistory();

    if (history.length === 0) {
        historyList.innerHTML = `
            <li class="history-item" style="color: var(--text-muted); font-style: italic;">
                No predictions yet. Submit a prediction to see your history.
            </li>
        `;
        return;
    }

    historyList.innerHTML = history.map(entry => `
        <li class="history-item">
            <div class="history-item__info">
                <span class="history-item__visa">${entry.visa_type} - ${entry.nationality}</span>
                <span class="history-item__date">${entry.date}</span>
            </div>
            <div class="history-item__days">${entry.predicted_days} days</div>
        </li>
    `).join('');
}

// Clear all history
function clearHistory() {
    if (confirm('Are you sure you want to clear your prediction history?')) {
        localStorage.removeItem(HISTORY_KEY);
        renderHistory();
    }
}

// ============================================
// MULTI-LANGUAGE SUPPORT (Basic)
// ============================================
const translations = {
    en: {
        title: 'Know Your Visa Processing Time',
        submit: 'Get My Estimate',
        reset: 'Try Another Estimate',
        history: 'Your Prediction History',
        trends: 'Monthly Processing Trends'
    },
    hi: {
        title: 'अपना वीज़ा प्रोसेसिंग समय जानें',
        submit: 'मेरा अनुमान प्राप्त करें',
        reset: 'एक और अनुमान आज़माएं',
        history: 'आपका पूर्वानुमान इतिहास',
        trends: 'मासिक प्रोसेसिंग रुझान'
    },
    es: {
        title: 'Conozca su tiempo de procesamiento de visa',
        submit: 'Obtener mi estimación',
        reset: 'Intentar otra estimación',
        history: 'Tu historial de predicciones',
        trends: 'Tendencias de procesamiento mensual'
    }
};

let currentLang = localStorage.getItem('visachronos_lang') || 'en';

function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('visachronos_lang', lang);
    // Note: Full implementation would update all text elements
    // This is a foundation for future internationalization
}

// ============================================
// CUSTOM VALIDATION (Premium UI)
// ============================================
function initCustomValidation() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    // Add custom validation messages
    const selects = form.querySelectorAll('select[required]');
    const inputs = form.querySelectorAll('input[required]');

    // Disable browser default validation UI
    form.setAttribute('novalidate', 'true');

    // Add validation message elements
    [...selects, ...inputs].forEach(field => {
        const group = field.closest('.form-group');
        if (group && !group.querySelector('.validation-message')) {
            const msg = document.createElement('div');
            msg.className = 'validation-message';
            msg.textContent = '⚠ Please fill this field';
            group.appendChild(msg);
        }

        // Listen for changes
        field.addEventListener('blur', function () {
            validateField(this);
        });

        field.addEventListener('change', function () {
            validateField(this);
        });
    });

    // Custom form submit validation
    form.addEventListener('submit', function (e) {
        let isValid = true;

        [...selects, ...inputs].forEach(field => {
            if (!validateField(field)) {
                isValid = false;
            }
        });

        if (!isValid) {
            e.preventDefault();
            // Shake animation on invalid fields
            const errorFields = form.querySelectorAll('.form-group.error');
            errorFields.forEach(field => {
                field.style.animation = 'shake 0.4s ease';
                setTimeout(() => field.style.animation = '', 400);
            });
        }
    });
}

function validateField(field) {
    const group = field.closest('.form-group');
    if (!group) return true;

    const isSelect = field.tagName === 'SELECT';
    const isEmpty = isSelect
        ? (field.value === '' || field.selectedIndex === 0)
        : field.value.trim() === '';

    if (field.required && isEmpty) {
        group.classList.add('error');
        const msg = group.querySelector('.validation-message');
        if (msg) {
            msg.textContent = isSelect
                ? '⚠ Please select an option'
                : '⚠ This field is required';
        }
        return false;
    } else {
        group.classList.remove('error');
        return true;
    }
}

// Add shake animation
const shakeKeyframes = `
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    20% { transform: translateX(-8px); }
    40% { transform: translateX(8px); }
    60% { transform: translateX(-4px); }
    80% { transform: translateX(4px); }
}
`;
if (!document.getElementById('shake-style')) {
    const style = document.createElement('style');
    style.id = 'shake-style';
    style.textContent = shakeKeyframes;
    document.head.appendChild(style);
}

// ============================================
// CUSTOM STYLED DROPDOWN
// ============================================
const countryFlags = {
    'Afghanistan': '🇦🇫', 'Argentina': '🇦🇷', 'Australia': '🇦🇺', 'Austria': '🇦🇹',
    'Bangladesh': '🇧🇩', 'Belgium': '🇧🇪', 'Brazil': '🇧🇷', 'Canada': '🇨🇦',
    'Chile': '🇨🇱', 'China': '🇨🇳', 'Colombia': '🇨🇴', 'Czech Republic': '🇨🇿',
    'Denmark': '🇩🇰', 'Egypt': '🇪🇬', 'Finland': '🇫🇮', 'France': '🇫🇷',
    'Germany': '🇩🇪', 'Greece': '🇬🇷', 'Hungary': '🇭🇺', 'India': '🇮🇳',
    'Indonesia': '🇮🇩', 'Iran': '🇮🇷', 'Iraq': '🇮🇶', 'Ireland': '🇮🇪',
    'Israel': '🇮🇱', 'Italy': '🇮🇹', 'Japan': '🇯🇵', 'Kenya': '🇰🇪',
    'Malaysia': '🇲🇾', 'Mexico': '🇲🇽', 'Morocco': '🇲🇦', 'Nepal': '🇳🇵',
    'Netherlands': '🇳🇱', 'New Zealand': '🇳🇿', 'Nigeria': '🇳🇬', 'Norway': '🇳🇴',
    'Pakistan': '🇵🇰', 'Peru': '🇵🇪', 'Philippines': '🇵🇭', 'Poland': '🇵🇱',
    'Portugal': '🇵🇹', 'Romania': '🇷🇴', 'Russia': '🇷🇺', 'Saudi Arabia': '🇸🇦',
    'Singapore': '🇸🇬', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'Spain': '🇪🇸',
    'Sri Lanka': '🇱🇰', 'Sweden': '🇸🇪', 'Switzerland': '🇨🇭', 'Thailand': '🇹🇭',
    'Turkey': '🇹🇷', 'UAE': '🇦🇪', 'UK': '🇬🇧', 'USA': '🇺🇸',
    'Ukraine': '🇺🇦', 'Vietnam': '🇻🇳', 'United States': '🇺🇸', 'United Kingdom': '🇬🇧'
};

function initCustomDropdowns() {
    const selects = document.querySelectorAll('select[data-custom-select]');

    selects.forEach(select => {
        createCustomDropdown(select);
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.custom-select')) {
            document.querySelectorAll('.custom-select.open').forEach(el => {
                el.classList.remove('open');
            });
        }
    });
}

function createCustomDropdown(nativeSelect) {
    const wrapper = document.createElement('div');
    wrapper.className = 'custom-select';

    // Detect if this is a country/nationality dropdown
    const selectId = nativeSelect.id || '';
    const isCountrySelect = selectId.includes('nationality') || selectId.includes('country');
    wrapper.dataset.hasFlags = isCountrySelect;

    // Get options from native select
    const options = Array.from(nativeSelect.options).slice(1); // Skip placeholder
    const placeholder = nativeSelect.options[0]?.text || 'Select...';

    // Create trigger button
    const trigger = document.createElement('div');
    trigger.className = 'custom-select__trigger';
    trigger.innerHTML = `
        <span class="custom-select__value">
            <span class="custom-select__placeholder">${placeholder}</span>
        </span>
        <svg class="custom-select__arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 9l6 6 6-6"/>
        </svg>
    `;

    // Create dropdown panel
    const dropdown = document.createElement('div');
    dropdown.className = 'custom-select__dropdown';

    // Create search box
    const searchBox = document.createElement('div');
    searchBox.className = 'custom-select__search';
    searchBox.innerHTML = `
        <svg class="custom-select__search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/>
            <path d="M21 21l-4.35-4.35"/>
        </svg>
        <input type="text" class="custom-select__search-input" placeholder="Search...">
    `;

    // Create options list
    const optionsList = document.createElement('div');
    optionsList.className = 'custom-select__options';

    options.forEach(opt => {
        const optionEl = document.createElement('div');
        optionEl.className = 'custom-select__option';
        optionEl.dataset.value = opt.value;

        // Only add flag for country dropdowns
        if (isCountrySelect) {
            const flag = countryFlags[opt.text] || countryFlags[opt.value] || '';
            optionEl.innerHTML = `
                ${flag ? `<span class="custom-select__option-flag">${flag}</span>` : ''}
                <span class="custom-select__option-text">${opt.text}</span>
            `;
        } else {
            optionEl.innerHTML = `<span class="custom-select__option-text">${opt.text}</span>`;
        }

        optionEl.addEventListener('click', () => {
            const flag = isCountrySelect ? (countryFlags[opt.text] || countryFlags[opt.value] || '') : '';
            selectOption(wrapper, nativeSelect, opt.value, opt.text, flag, isCountrySelect);
        });

        optionsList.appendChild(optionEl);
    });

    // Add no results message
    const noResults = document.createElement('div');
    noResults.className = 'custom-select__no-results';
    noResults.textContent = 'No results found';
    noResults.style.display = 'none';
    optionsList.appendChild(noResults);

    dropdown.appendChild(searchBox);
    dropdown.appendChild(optionsList);

    wrapper.appendChild(trigger);
    wrapper.appendChild(dropdown);

    // Hide native select but keep for form submission
    nativeSelect.classList.add('native-select');
    nativeSelect.parentNode.insertBefore(wrapper, nativeSelect);
    wrapper.appendChild(nativeSelect);

    // Toggle dropdown on trigger click
    trigger.addEventListener('click', (e) => {
        e.stopPropagation();

        // Close other dropdowns
        document.querySelectorAll('.custom-select.open').forEach(el => {
            if (el !== wrapper) el.classList.remove('open');
        });

        wrapper.classList.toggle('open');

        if (wrapper.classList.contains('open')) {
            const searchInput = searchBox.querySelector('.custom-select__search-input');
            setTimeout(() => searchInput.focus(), 100);
        }
    });

    // Search functionality
    const searchInput = searchBox.querySelector('.custom-select__search-input');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        let hasResults = false;

        optionsList.querySelectorAll('.custom-select__option').forEach(opt => {
            const text = opt.querySelector('.custom-select__option-text').textContent.toLowerCase();
            const matches = text.includes(query);
            opt.classList.toggle('hidden', !matches);
            if (matches) hasResults = true;
        });

        noResults.style.display = hasResults ? 'none' : 'block';
    });

    searchInput.addEventListener('click', (e) => e.stopPropagation());

    // Set initial value if native select has one
    if (nativeSelect.value) {
        const selectedOpt = nativeSelect.options[nativeSelect.selectedIndex];
        if (selectedOpt && selectedOpt.value) {
            const flag = isCountrySelect ? (countryFlags[selectedOpt.text] || countryFlags[selectedOpt.value] || '') : '';
            selectOption(wrapper, nativeSelect, selectedOpt.value, selectedOpt.text, flag, isCountrySelect, false);
        }
    }
}

function selectOption(wrapper, nativeSelect, value, text, flag, isCountrySelect, closeDropdown = true) {
    // Update native select
    nativeSelect.value = value;
    nativeSelect.dispatchEvent(new Event('change', { bubbles: true }));

    // Update trigger display
    const valueEl = wrapper.querySelector('.custom-select__value');
    if (flag && isCountrySelect) {
        valueEl.innerHTML = `
            <span class="custom-select__option-flag">${flag}</span>
            <span>${text}</span>
        `;
    } else {
        valueEl.innerHTML = `<span>${text}</span>`;
    }

    // Mark option as selected
    wrapper.querySelectorAll('.custom-select__option').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.value === value);
    });

    // Close dropdown
    if (closeDropdown) {
        wrapper.classList.remove('open');
    }

    // Clear search
    const searchInput = wrapper.querySelector('.custom-select__search-input');
    if (searchInput) {
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input'));
    }
}

// ============================================
// RESET FORM FUNCTION
// ============================================
function resetForm() {
    // Get the form element
    const form = document.getElementById('prediction-form');
    if (form) {
        form.reset();
    }

    // Reset all custom dropdowns to their initial state
    document.querySelectorAll('.custom-select').forEach(wrapper => {
        const nativeSelect = wrapper.querySelector('select');
        if (nativeSelect) {
            // Reset to first option (usually placeholder)
            nativeSelect.selectedIndex = 0;

            // Update the custom dropdown display
            const valueEl = wrapper.querySelector('.custom-select__value');
            const placeholder = nativeSelect.options[0]?.text || 'Select...';
            valueEl.innerHTML = `<span class="custom-select__placeholder">${placeholder}</span>`;

            // Remove selected state from all options
            wrapper.querySelectorAll('.custom-select__option').forEach(opt => {
                opt.classList.remove('selected');
            });
        }
    });

    // Hide results panel if visible
    const resultsPanel = document.getElementById('results-panel');
    if (resultsPanel) {
        resultsPanel.style.display = 'none';
        resultsPanel.classList.remove('active');
    }

    // Scroll to top of form
    if (form) {
        form.scrollIntoView({ behavior: 'smooth' });
    }
}

// ============================================
// BACK TO TOP BUTTON
// ============================================
function initBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;

    window.addEventListener('scroll', function () {
        if (window.scrollY > 300) {
            btn.classList.add('visible');
        } else {
            btn.classList.remove('visible');
        }
    });

    btn.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// ============================================
// ANIMATED COUNTERS (Stats Bar)
// ============================================
function initAnimatedCounters() {
    const statValues = document.querySelectorAll('.stat__value');
    if (statValues.length === 0) return;

    const animateCounter = (el) => {
        const text = el.textContent.trim();
        // Extract number and suffix (e.g. "2,000+" → 2000, "+", or "77%" → 77, "%")
        const match = text.match(/^([\d,]+)(.*)/);
        if (!match) return;

        const target = parseInt(match[1].replace(/,/g, ''));
        const suffix = match[2]; // "+", "%", etc.
        const hasComma = match[1].includes(',');
        const duration = 1500;
        const start = performance.now();

        el.textContent = '0' + suffix;

        const step = (now) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out curve for smooth deceleration
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(target * eased);

            if (hasComma) {
                el.textContent = current.toLocaleString() + suffix;
            } else {
                el.textContent = current + suffix;
            }

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };

        requestAnimationFrame(step);
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const stats = entry.target.querySelectorAll('.stat__value');
                stats.forEach((stat, i) => {
                    setTimeout(() => animateCounter(stat), i * 150);
                });
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    const statsBar = document.querySelector('.stats-bar');
    if (statsBar) observer.observe(statsBar);
}

// ============================================
// AI-GENERATED NATURAL LANGUAGE SUMMARY
// ============================================
function generateAISummary(data, formData) {
    const section = document.getElementById('ai-summary-section');
    const textEl = document.getElementById('ai-summary-text');
    if (!section || !textEl || !formData) return;

    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    const month = monthNames[(formData.application_month || 1) - 1];
    const visa = formData.visa_type || 'Tourist';
    const country = formData.nationality || 'Unknown';
    const days = data.predicted_days;
    const minD = data.min_days;
    const maxD = data.max_days;
    const risk = data.risk_level;
    const approval = data.approval_percentage;
    const countryAvg = data.country_average;
    const visaAvg = data.visa_type_average;

    let comparison = '';
    if (days < visaAvg) {
        const pct = Math.round(((visaAvg - days) / visaAvg) * 100);
        comparison = `This is <strong>${pct}% faster</strong> than the average ${visa} visa processing time of ${visaAvg} days.`;
    } else if (days > visaAvg) {
        const pct = Math.round(((days - visaAvg) / visaAvg) * 100);
        comparison = `This is <strong>${pct}% slower</strong> than the average ${visa} visa processing time of ${visaAvg} days.`;
    } else {
        comparison = `This matches the average ${visa} visa processing time.`;
    }

    let seasonNote = '';
    const peakMonths = [10, 11, 3, 4]; // Oct, Nov, Mar, Apr
    if (peakMonths.includes(formData.application_month)) {
        seasonNote = ` Applying in <strong>${month}</strong> falls within peak processing season, which may contribute to slightly longer wait times.`;
    } else {
        seasonNote = ` Applying in <strong>${month}</strong> is outside peak season, which generally results in faster processing.`;
    }

    let riskNote = '';
    if (risk === 'Low') {
        riskNote = ' Your application has been assessed as <strong>low risk</strong>, indicating strong documentation and favorable profile characteristics.';
    } else if (risk === 'Medium') {
        riskNote = ' Your application has a <strong>moderate risk</strong> profile. Consider ensuring all supporting documents are complete for the best outcome.';
    } else {
        riskNote = ' Your application shows <strong>elevated risk factors</strong>. We recommend double-checking all documentation and financial proofs before submission.';
    }

    const summary = `Your <strong>${visa}</strong> visa application from <strong>${country}</strong> has an estimated processing time of <strong>${days} days</strong> (range: ${minD}–${maxD} days), with a <strong>${approval}% approval likelihood</strong>. ${comparison}${seasonNote}${riskNote} Compared to the country average of ${countryAvg} days for ${country}, your estimate is ${days < countryAvg ? 'better' : days > countryAvg ? 'above average' : 'on par'}.`;

    textEl.innerHTML = summary;
    section.style.display = 'block';
}

// ============================================
// WATERFALL CHART — FEATURE CONTRIBUTION
// ============================================
function renderWaterfallChart(data) {
    const section = document.getElementById('waterfall-section');
    const chart = document.getElementById('waterfall-chart');
    if (!section || !chart) return;

    const baseline = 8.2; // overall average
    const predicted = data.predicted_days;

    // Calculate factor contributions based on the prediction factors
    const factors = [
        { label: 'Baseline Avg', value: 0, isBaseline: true },
        { label: 'Visa Type', value: data.visa_type_average - baseline },
        { label: 'Nationality', value: data.country_average - baseline },
        { label: 'Season', value: data.is_peak_season ? 1.2 : -0.8 },
        { label: 'Documents', value: data.factors?.documents_complete === 'Complete' ? -0.6 : 1.4 },
        { label: 'Express', value: data.factors?.express === 'Yes' ? -2.5 : 0 },
        { label: 'Sponsor', value: data.factors?.sponsor === 'Yes' ? -0.5 : 0.3 },
        { label: 'Risk Level', value: data.risk_score > 60 ? 1.0 : data.risk_score > 30 ? 0.2 : -0.5 }
    ];

    // Normalize so they sum to predicted - baseline
    const rawSum = factors.slice(1).reduce((s, f) => s + f.value, 0);
    const targetDiff = predicted - baseline;
    const scale = rawSum !== 0 ? targetDiff / rawSum : 1;
    factors.slice(1).forEach(f => f.value = Math.round(f.value * scale * 10) / 10);

    const maxAbsVal = Math.max(3, ...factors.map(f => Math.abs(f.value)));

    let html = '';

    // Baseline row
    html += `<div class="waterfall-row">
        <div class="waterfall-label">Baseline</div>
        <div class="waterfall-bar-container">
            <div class="waterfall-bar-baseline" style="left:50%"></div>
        </div>
        <div class="waterfall-value">${baseline.toFixed(1)}d</div>
    </div>`;

    // Factor rows
    factors.slice(1).forEach(f => {
        if (f.value === 0) return;
        const isPos = f.value > 0;
        const barWidth = Math.min(45, (Math.abs(f.value) / maxAbsVal) * 45);
        const barLeft = isPos ? 50 : 50 - barWidth;
        const cls = isPos ? 'positive' : 'negative';
        const sign = isPos ? '+' : '';

        html += `<div class="waterfall-row">
            <div class="waterfall-label">${f.label}</div>
            <div class="waterfall-bar-container">
                <div class="waterfall-bar-baseline" style="left:50%"></div>
                <div class="waterfall-bar waterfall-bar--${cls}" style="left:${barLeft}%;width:${barWidth}%"></div>
            </div>
            <div class="waterfall-value waterfall-value--${cls}">${sign}${f.value.toFixed(1)}d</div>
        </div>`;
    });

    // Total row
    const totalBarWidth = Math.min(90, (predicted / (baseline + maxAbsVal)) * 90);
    html += `<div class="waterfall-row waterfall-row--total">
        <div class="waterfall-label">Your Estimate</div>
        <div class="waterfall-bar-container">
            <div class="waterfall-bar waterfall-bar--total" style="left:0;width:${totalBarWidth}%"></div>
        </div>
        <div class="waterfall-value waterfall-value--total">${predicted.toFixed(1)}d</div>
    </div>`;

    chart.innerHTML = html;
    section.style.display = 'block';
}

// ============================================
// WHAT-IF SCENARIO ANALYSIS
// ============================================
async function runWhatIfAnalysis(formData, currentResult) {
    const section = document.getElementById('whatif-section');
    const grid = document.getElementById('whatif-grid');
    if (!section || !grid || !formData) return;

    section.style.display = 'block';
    grid.innerHTML = '<div class="whatif-loading"><div class="whatif-spinner"></div><span>Running AI scenario analysis...</span></div>';

    const scenarios = [
        {
            name: 'Current Application',
            desc: 'Your submitted parameters',
            icon: '📋',
            iconClass: 'current',
            data: { ...formData },
            days: currentResult.predicted_days,
            isCurrent: true
        },
        {
            name: formData.express_processing ? 'Without Express' : 'With Express Processing',
            desc: formData.express_processing ? 'Standard processing speed' : 'Fast-track your application',
            icon: '⚡',
            iconClass: 'express',
            data: { ...formData, express_processing: !formData.express_processing }
        },
        {
            name: formData.documents_complete ? 'Incomplete Documents' : 'Complete Documents',
            desc: formData.documents_complete ? 'Missing some documents' : 'All documents submitted',
            icon: '📄',
            iconClass: 'docs',
            data: { ...formData, documents_complete: !formData.documents_complete }
        },
        {
            name: 'Opposite Season',
            desc: `Apply in ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][((formData.application_month + 5) % 12)]} instead`,
            icon: '🌦️',
            iconClass: 'season',
            data: { ...formData, application_month: ((formData.application_month + 5) % 12) + 1 }
        }
    ];

    // Fetch predictions for non-current scenarios
    const promises = scenarios.slice(1).map(async (s) => {
        try {
            const resp = await fetch(API_BASE + '/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(s.data)
            });
            if (!resp.ok) throw new Error('Failed');
            const result = await resp.json();
            s.days = result.predicted_days;
        } catch {
            s.days = null;
        }
    });

    await Promise.all(promises);

    const baseDays = currentResult.predicted_days;
    let html = '';

    scenarios.forEach(s => {
        if (s.days === null) return;
        const diff = s.days - baseDays;
        const diffText = s.isCurrent ? 'baseline' : (diff > 0 ? `+${diff.toFixed(1)}d slower` : diff < 0 ? `${diff.toFixed(1)}d faster` : 'same');
        const cardClass = s.isCurrent ? 'whatif-card--current' : (diff < 0 ? 'whatif-card--better' : diff > 0 ? 'whatif-card--worse' : '');
        const daysClass = s.isCurrent ? 'whatif-result__days--current' : (diff < 0 ? 'whatif-result__days--better' : diff > 0 ? 'whatif-result__days--worse' : '');
        const diffColor = s.isCurrent ? 'color:var(--accent)' : (diff < 0 ? 'color:#22c55e' : 'color:#ef4444');

        html += `<div class="whatif-card ${cardClass}">
            <div class="whatif-info">
                <h4>${s.name}</h4>
                <p>${s.desc}</p>
            </div>
            <div class="whatif-result">
                <div class="whatif-result__days ${daysClass}">${s.days.toFixed(1)}d</div>
                <div class="whatif-result__diff" style="${diffColor}">${diffText}</div>
            </div>
        </div>`;
    });

    grid.innerHTML = html;
}

// ============================================
// OPTIMAL MONTH RECOMMENDER
// ============================================
async function findOptimalMonth(formData) {
    const section = document.getElementById('optimal-month-section');
    const chartEl = document.getElementById('optimal-month-chart');
    const recEl = document.getElementById('optimal-month-recommendation');
    if (!section || !chartEl || !formData) return;

    section.style.display = 'block';
    chartEl.innerHTML = '<div class="whatif-loading"><div class="whatif-spinner"></div><span>Analyzing all 12 months...</span></div>';

    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const results = [];

    // Fetch predictions for all 12 months
    const promises = Array.from({ length: 12 }, (_, i) => {
        const monthData = { ...formData, application_month: i + 1 };
        return fetch(API_BASE + '/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(monthData)
        })
            .then(r => r.json())
            .then(data => { results[i] = data.predicted_days; })
            .catch(() => { results[i] = null; });
    });

    await Promise.all(promises);

    const validResults = results.filter(r => r !== null);
    if (validResults.length === 0) return;

    const minDays = Math.min(...validResults);
    const maxDays = Math.max(...validResults);
    const bestMonth = results.indexOf(minDays);
    const worstMonth = results.indexOf(maxDays);
    const currentMonth = (formData.application_month || 1) - 1;
    const range = maxDays - minDays || 1;

    let html = '';
    results.forEach((days, i) => {
        if (days === null) return;
        const height = 20 + ((days - minDays) / range) * 120; // invert: shorter = less days = shorter bar
        const invertedHeight = 20 + ((maxDays - days) / range) * 120; // better months get taller bars
        let fillClass = '';
        let labelClass = '';
        if (i === bestMonth) { fillClass = 'month-bar__fill--best'; labelClass = 'month-bar__label--best'; }
        else if (i === worstMonth) { fillClass = 'month-bar__fill--worst'; }
        else if (i === currentMonth) { fillClass = 'month-bar__fill--current'; labelClass = 'month-bar__label--current'; }

        html += `<div class="month-bar">
            <div class="month-bar__value">${days.toFixed(1)}</div>
            <div class="month-bar__fill ${fillClass}" style="height:${invertedHeight}px"></div>
            <div class="month-bar__label ${labelClass}">${monthNames[i]}${i === bestMonth ? ' ★' : ''}</div>
        </div>`;
    });

    chartEl.innerHTML = html;

    // Show recommendation
    if (recEl) {
        const saving = results[currentMonth] - minDays;
        const savingPct = Math.round((saving / results[currentMonth]) * 100);
        if (bestMonth === currentMonth) {
            recEl.innerHTML = `✅ Great news! <strong>${monthNames[bestMonth]}</strong> is already the optimal month to apply. You're on track for the fastest processing time of <strong>${minDays.toFixed(1)} days</strong>.`;
        } else {
            recEl.innerHTML = `💡 <strong>Recommendation:</strong> Applying in <strong>${monthNames[bestMonth]}</strong> could save you <strong>${saving.toFixed(1)} days</strong> (${savingPct}% faster) compared to your current month (${monthNames[currentMonth]}). Best: <strong>${minDays.toFixed(1)}d</strong> vs Current: <strong>${results[currentMonth].toFixed(1)}d</strong>.`;
        }
        recEl.style.display = 'block';
    }
}

// ============================================
// GEOPOLITICAL COUNTRY RISK WARNING
// ============================================
const COUNTRY_RISK_DATA = {
    // High Alert countries
    'Pakistan': {
        level: 'high',
        icon: '🔴',
        title: '⚠️ HIGH ALERT — India–Pakistan Conflict Zone',
        text: 'Due to active India–Pakistan tensions, visa processing for Pakistani nationals is experiencing <strong>severe delays (+15–30 days)</strong>. Airspace is restricted — flights between Pakistan and India are suspended. Dubai–India routes are also affected by missile threat alerts. <strong>Processing may be temporarily suspended.</strong>'
    },
    'Afghanistan': {
        level: 'high',
        icon: '🔴',
        title: '⚠️ HIGH ALERT — Conflict Zone',
        text: 'Afghan nationals may face <strong>significant processing delays (+10–25 days)</strong> due to ongoing regional instability and India–Pakistan airspace restrictions. Applications require enhanced security screening. Recommend express processing.'
    },
    'Israel': {
        level: 'high',
        icon: '🔴',
        title: '⚠️ HIGH ALERT — Israel–Iran–USA Conflict',
        text: 'Due to the ongoing Israel–Iran–USA conflict, Israeli nationals may experience <strong>processing delays (+10–20 days)</strong>. Middle East airspace disruptions affect Gulf flight routes. Enhanced security screening is in effect.'
    },
    'Iran': {
        level: 'high',
        icon: '🔴',
        title: '⚠️ HIGH ALERT — Iran Conflict Zone',
        text: 'Iranian nationals are subject to <strong>extended processing times (+15–30 days)</strong> due to the Israel–Iran–USA conflict and international sanctions. Gulf airspace is disrupted. Applications undergo rigorous security verification.'
    },
    // Elevated countries
    'Bangladesh': {
        level: 'elevated',
        icon: '🟡',
        title: '🟡 ELEVATED — Regional Tension Impact',
        text: 'Bangladeshi applicants may see <strong>moderate delays (+5–10 days)</strong> due to proximity to the India–Pakistan conflict zone and increased border scrutiny. Document completeness is critical.'
    },
    'Iraq': {
        level: 'elevated',
        icon: '🟡',
        title: '🟡 ELEVATED — Middle East Instability',
        text: 'Iraqi nationals may experience <strong>additional processing delays (+5–10 days)</strong> due to the ongoing Middle East conflict. Gulf flight routes are disrupted. Allow extra time.'
    },
    'UAE': {
        level: 'elevated',
        icon: '🟡',
        title: '🟡 ELEVATED — Gulf Route Disruptions',
        text: 'UAE-based applicants should note that <strong>Dubai–India flights are experiencing disruptions</strong> due to missile threat alerts. Processing is active but <strong>may take 3–7 extra days</strong> due to airspace diversions.'
    },
    'Turkey': {
        level: 'elevated',
        icon: '🟡',
        title: '🟡 ELEVATED — Regional Proximity',
        text: 'Turkish nationals may face <strong>moderate delays (+3–7 days)</strong> due to proximity to the Israel–Iran conflict zone. Flight routes through the region are affected.'
    },
    'Ukraine': {
        level: 'elevated',
        icon: '🟡',
        title: '🟡 ELEVATED — Ongoing Conflict',
        text: 'Ukrainian applicants may experience <strong>processing delays (+5–10 days)</strong> due to the ongoing Russia–Ukraine situation. Enhanced documentation verification is in effect.'
    },
    'Russia': {
        level: 'elevated',
        icon: '🟡',
        title: '🟡 ELEVATED — Diplomatic Tensions',
        text: 'Russian nationals may see <strong>extended processing times (+5–10 days)</strong> due to diplomatic tensions and sanctions. Applications undergo additional verification.'
    }
};

function initCountryRiskWarning() {
    const nationalitySelect = document.getElementById('nationality');
    if (!nationalitySelect) return;

    nationalitySelect.addEventListener('change', function () {
        const country = this.value;
        const warningEl = document.getElementById('country-risk-warning');
        if (!warningEl) return;

        const risk = COUNTRY_RISK_DATA[country];

        if (risk) {
            document.getElementById('crw-icon').textContent = risk.icon;
            document.getElementById('crw-title').textContent = risk.title;
            document.getElementById('crw-text').innerHTML = risk.text;
            warningEl.className = `country-risk-warning country-risk-warning--${risk.level}`;
            warningEl.style.display = 'flex';
            // Animate entry
            warningEl.style.animation = 'none';
            warningEl.offsetHeight; // trigger reflow
            warningEl.style.animation = 'warningSlideIn 0.4s ease-out';
        } else {
            warningEl.style.display = 'none';
        }
    });
}

// ============================================
// INITIALIZE
// ============================================
document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initScrollReveal();
    initBackToTop();
    initAnimatedCounters();
    setDefaultMonth();
    renderHistory(); // Load prediction history
    initCustomValidation(); // Custom validation UI
    initCustomDropdowns(); // Custom styled dropdowns
    initCountryRiskWarning(); // Geopolitical country risk warnings
});
