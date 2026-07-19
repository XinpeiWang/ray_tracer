#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QThread>
#include <QLineEdit>
#include <QProcess>

class RenderThread : public QThread {
	Q_OBJECT

public:
	RenderThread(QObject *parent = nullptr);
	void setParameters(bool useGPU, int width, int height, int samples, int maxDepth, const QString &outputPath = QString());
	void stopRender();

protected:
	void run() override;

signals:
	void progressUpdate(int percentage);
	void renderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void logMessage(const QString &message);

private:
	bool m_useGPU;
	int m_width;
	int m_height;
	int m_samples;
	int m_maxDepth;
	QString m_outputPath;
	QProcess *m_renderProcess;
};

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = nullptr);
	~MainWindow();

private slots:
	void onRenderClicked();
	void onStopClicked();
	void onQualityPresetChanged(int index);
	void onProgressUpdate(int percentage);
	void onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void onLogMessage(const QString &message);

private:
	void setupUI();
	void createBasicTab();
	void createAdvancedTab();
	void applyDarkTheme();

	// UI Components
	QTabWidget *m_tabWidget;

	// Basic Tab
	QComboBox *m_renderModeCombo;
	QComboBox *m_qualityPresetCombo;
	QComboBox *m_resolutionCombo;
	QLineEdit *m_outputPathEdit;
	QPushButton *m_browseButton;
	QPushButton *m_renderButton;
	QPushButton *m_stopButton;
	QProgressBar *m_progressBar;
	QLabel *m_statusLabel;

	// Advanced Tab
	QSpinBox *m_widthSpinBox;
	QSpinBox *m_heightSpinBox;
	QSpinBox *m_samplesSpinBox;
	QSpinBox *m_maxDepthSpinBox;

	// Render thread
	RenderThread *m_renderThread;

	// State
	bool m_isRendering;
};

#endif // MAINWINDOW_H
