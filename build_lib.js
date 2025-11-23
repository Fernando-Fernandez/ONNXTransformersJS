const fs = require('fs');
const path = require('path');

const inputPath = path.join(__dirname, 'public', 'transformers.iife.js');
const outputPath = path.join(__dirname, 'public', 'transformers_lib.js');

try {
    const content = fs.readFileSync(inputPath, 'utf8');
    // Escape backticks and backslashes to be safe inside a template literal
    // Also polyfill import.meta which esbuild leaves empty in IIFE
    // And suppress "Automatic publicPath" error
    // And patch esbuild's import_meta variable
    // And patch location.origin for blob workers
    const escapedContent = content
        .replace(/\s*var import_meta = \{\};/g, 'var import_meta = { url: self.transformersBaseUrl || self.location.href };')
        .replace(/import\.meta/g, '({ url: self.transformersBaseUrl || self.location.href })')
        .replace(/typeof location > "u" \? void 0 : location\.origin/g, '(typeof location > "u" || location.origin === "null") ? void 0 : location.origin')
        .replace(/throw new Error\("Automatic publicPath is not supported in this browser"\)/g, 'console.warn("Automatic publicPath warning suppressed")')
        .replace(/\\/g, '\\\\')
        .replace(/`/g, '\\`')
        .replace(/\${/g, '\\${');

    const output = `const TRANSFORMERS_LIB = \`${escapedContent}\`;`;

    fs.writeFileSync(outputPath, output);
    console.log('Successfully created transformers_lib.js');
} catch (err) {
    console.error('Error processing file:', err);
    process.exit(1);
}
