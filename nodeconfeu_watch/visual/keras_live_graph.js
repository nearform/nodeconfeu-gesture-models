;(function () {
  const margin = {top: 10, right: 10, bottom: 10, left: 35}
  const facetWidth = 30;
  const legendHeight = 40;
  const xAxisHeight = 30;
  const xLabelHeight = 20;

  class SubGraph {
    constructor({ container, topName, name, lineSetup, outerWidth, outerHeight, drawXAxis, xlimEpochs, ylimFixed }) {
      this.width = outerWidth - facetWidth - margin.left - margin.right;
      this.height = outerHeight - margin.top - margin.bottom;
      this.xlimEpochs = xlimEpochs;
      this.ylimFixed = ylimFixed;

      this.lineSetup = lineSetup;
      this.container = container;
      this.graph = this.container.append('g')
        .attr('transform',
              'translate(' + margin.left + ',' + margin.top + ')');

      this.facet = this.container.append('g')
        .classed('facet', true)
        .attr('transform', `translate(${margin.left + this.width}, ${margin.top})`);
      this.facet.append('rect')
        .classed('facet-background', true)
        .attr('width', facetWidth)
        .attr('height', this.height);
      const facetTextPath = this.facet.append('path')
        .attr('d', `M10,0 V${this.height}`)
        .attr('id', `ar-training-graph-${topName}-${name}-facet-text`);
      const facetText = this.facet.append('text')
        .append('textPath')
        .attr('startOffset', '50%')
        .attr('href', `#ar-training-graph-${topName}-${name}-facet-text`)
        .attr('text-anchor', 'middle')
        .text(name);
      facetText.node()
        .setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', `#ar-training-graph-${topName}-${name}-facet-text`);

      // Create background
      this.background = this.graph.append("rect")
        .attr("class", "background")
        .attr("height", this.height)
        .attr("width", this.width);

      // define scales
      this.yScale = d3.scaleLinear()
        .domain(this.ylimFixed ? this.ylimFixed : [0, 1])
        .range([this.height, 0]);

      this.xScale = d3.scaleLinear()
        .domain(this.xlimEpochs)
        .range([0, this.width]);

      // create grid
      this.xGrid = d3.axisBottom(this.xScale)
        .ticks(17)
        .tickSize(-this.height);
      this.xGridElement = this.graph.append("g")
          .attr("class", "grid")
          .attr("transform", "translate(0," + this.height + ")");

      const xTicksMajors = this.xScale.ticks(4);
      this.xGridElement.call(this.xGrid);
      this.xGridElement
        .selectAll('.tick')
        .classed('minor', function (d) {
          return !xTicksMajors.includes(d);
        })

      this.yGrid = d3.axisLeft(this.yScale)
        .ticks(8);
      this.yGridElement = this.graph.append("g")
          .attr("class", "grid");

      // define axis
      this.xAxis = d3.axisBottom(this.xScale)
        .ticks(9);
      this.xAxisElement = this.graph.append('g')
        .attr("class", "axis")
        .classed('hide-axis', !drawXAxis)
        .attr('transform', 'translate(0,' + this.height + ')');

      this.yAxis = d3.axisLeft(this.yScale)
        .ticks(4);
      this.yAxisElement = this.graph.append('g')
        .attr("class", "axis");
      this.xAxisElement.call(this.xAxis);

      // Define drawer functions and line elements
      this.lineDrawers = [];
      this.lineElements = [];
      const self = this;
      for (let i = 0; i < lineSetup.length; i++) {
        const lineDrawer = d3.line()
            .x((d) => self.xScale(d.epoch))
            .y((d) => this.yScale(d.value));

        this.lineDrawers.push(lineDrawer);

        const lineElement = this.graph.append('path')
            .attr('class', 'line')
            .attr('stroke', lineSetup[i].color);

        this.lineElements.push(lineElement);
      }
    }

    setData (data) {
      // Update domain of scales
      let yLimMax = 0;
      for (let i = 0; i < this.lineSetup.length; i++) {
        const lineData = data[this.lineSetup[i].name];
        yLimMax = Math.max(
          yLimMax,
          d3.max(lineData.map((d) => d.value))
        );

        this.lineElements[i].data([lineData]);
      }

      if (this.ylimFixed === null) {
        this.yScale.domain([-0.05*yLimMax, 1.05*yLimMax]);
      }
    }

    draw() {
      // update grid
      this.yGrid.tickSize(-this.width);
      const yTicksMajors = this.yScale.ticks(4);
      this.yGridElement.call(this.yGrid);
      this.yGridElement
        .selectAll('.tick')
        .classed('minor', function (d) {
          return !yTicksMajors.includes(d);
        })

      // update axis
      this.yAxisElement.call(this.yAxis);

      // update lines
      for (let i = 0; i < this.lineSetup.length; i++) {
        this.lineElements[i].attr('d', this.lineDrawers[i]);
      }
    }
  }

  class TrainingGraph {
    constructor({ container, name, height, width, epochs }) {
      const lineSetup = [
        {name: 'Train', color: "#F8766D"},
        {name: 'Validation', color: "#00BFC4"}
      ];

      const graphHeight = (height - legendHeight - xLabelHeight - xAxisHeight) / 2;
      const innerWidth = width - margin.left - margin.right - facetWidth;

      this._container = d3.select(container)
        .classed('ar-training-graph', true)
        .attr('height', height)
        .attr('xmlns:xlink', 'http://www.w3.org/1999/xlink');

      this._accuracy = new SubGraph({
        container: this._container.append('g'),
        topName: name,
        name: 'accuracy',
        lineSetup: lineSetup,
        outerHeight: graphHeight,
        outerWidth: width,
        drawXAxis: false,
        xlimEpochs: [-0.01*epochs, 1.01*epochs],
        ylimFixed: [-5, 105]
      });
      this._loss = new SubGraph({
        container: this._container.append('g')
          .attr('transform', `translate(0, ${graphHeight})`),
        topName: name,
        name: 'loss',
        lineSetup: lineSetup,
        outerHeight: graphHeight,
        outerWidth: width,
        drawXAxis: true,
        xlimEpochs: [-0.01*epochs, 1.01*epochs],
        ylimFixed: null
      });

      this._labels = this._container
        .append('g')
        .attr('transform', `translate(0, ${margin.top})`);

      const combinedOuterHeight = height - legendHeight;
      this._xLabel = this._labels.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', innerWidth / 2 + margin.left)
        .attr('y', combinedOuterHeight - xAxisHeight)
        .text('epoch');

      this._legend = this._container
        .append('g')
        .classed('legned', true)
        .attr('transform', `translate(${margin.left}, ${combinedOuterHeight})`);
      this._legendOfsset = this._legend.append('g');

      let currentOffset = 0;
      for (const {name, color} of lineSetup) {
        this._legendOfsset.append('rect')
          .attr('width', 25)
          .attr('height', 25)
          .attr('x', currentOffset);
        this._legendOfsset.append('line')
          .attr('x1', currentOffset + 2)
          .attr('x2', currentOffset + 25 - 2)
          .attr('y1', 25/2)
          .attr('y2', 25/2)
          .attr('stroke', color);

        const textNode = this._legendOfsset.append('text')
          .attr('x', currentOffset + 30)
          .attr('y', 19)
          .text(name);
        const textWidth = textNode.node().getComputedTextLength();
        currentOffset += 30 + textWidth + 20;
      }
      this._legendWidth = currentOffset - 20;

      this._legendOfsset
        .attr('transform', `translate(${(innerWidth - this._legendWidth) / 2}, 0)`);
    }

    getGraphWidth () {
      return this.width;
    }

    setData(data) {
      this._accuracy.setData(data.accuracy);
      this._loss.setData(data.loss);
    }

    draw() {
      this._accuracy.draw();
      this._loss.draw();
    }
  }

  class TrainingData {
    constructor() {
      this.accuracy = {
        'Train': [],
        'Validation': []
      };
      this.loss = {
        'Train': [],
        'Validation': []
      };
    }

    append(row) {
      this.accuracy['Train'].push({
        epoch: row.epoch, value: row['train.accuracy']
      })
      this.accuracy['Validation'].push({
        epoch: row.epoch, value: row['validation.accuracy']
      })
      this.loss['Train'].push({
        epoch: row.epoch, value: row['train.loss']
      })
      this.loss['Validation'].push({
        epoch: row.epoch, value: row['validation.loss']
      })
    }
  }

  window.setupLearningGraph = function ({id, epochs, height, width}) {
    const data = new TrainingData();
    const graph = new TrainingGraph({
      container: document.getElementById(id),
      name: id,
      epochs: epochs,
      height: height,
      width: width
    });

    let waitingForDrawing = false;
    function drawer() {
      waitingForDrawing = false;
      graph.setData(data);
      graph.draw();
    }

    window.appendLearningGraphData = function(rows) {
      for (const row of rows) {
        data.append({
          epoch: +row.epoch,
          'train.accuracy': +row['sparse_categorical_accuracy'] * 100,
          'validation.accuracy': +row['val_sparse_categorical_accuracy'] * 100,
          'train.loss': +row['loss'],
          'validation.loss': +row['val_loss']
        });
      }

      if (!waitingForDrawing) {
        waitingForDrawing = true;
        window.requestAnimationFrame(drawer);
      }
    };
  };
})();