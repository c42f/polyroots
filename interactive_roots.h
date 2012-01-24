#include <QtGui/QWidget>
#include <QtGui/QImage>

class RootViewWidget : public QWidget
{
    Q_OBJECT
    public:
        RootViewWidget();

    protected:
        void keyPressEvent(QKeyEvent* event);
        void paintEvent(QPaintEvent* event);
        void resizeEvent(QResizeEvent* event);
        void mousePressEvent(QMouseEvent* event);
        void mouseMoveEvent(QMouseEvent* event);

        //QSize sizeHint() const;

    private:
        void renderImage();

        QPoint m_lastPos;
        QImage m_image;
        int m_degree;
        QRectF m_bbox;
};

