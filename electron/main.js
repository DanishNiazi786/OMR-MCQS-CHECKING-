const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
    },
    titleBarStyle: 'default',
    icon: path.join(__dirname, 'assets/icon.png'),
  });

  const startUrl = isDev 
    ? 'http://localhost:5173' 
    : `file://${path.join(__dirname, '../dist/index.html')}`;
    
  mainWindow.loadURL(startUrl);

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers for scanner integration
ipcMain.handle('scanner-list', async () => {
  try {
    // Scanner integration would go here
    return [{ id: 'scanner1', name: 'Default Scanner' }];
  } catch (error) {
    console.error('Scanner error:', error);
    return [];
  }
});

ipcMain.handle('start-scan', async (event, config) => {
  try {
    // TWAIN scanner integration
    return { success: true, message: 'Scan started' };
  } catch (error) {
    return { success: false, error: error.message };
  }
});